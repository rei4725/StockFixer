"""build_execution_inputs の枝を1つずつ直接確かめる（#1919）。

期待値はすべて手で計算した値で書く。ラグ 1 の exec_avg_vol は「前の足までの平均」
（先頭は 0）であり、当日の足の平均（avg_vol）とは各足で違う値になる入力を使う。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.execution_inputs import build_execution_inputs

_IDX = pd.date_range("2026-01-05", periods=4, freq="D")


def _df(**cols: list[float]) -> pd.DataFrame:
    base = {"Close": [10.0, 11.0, 12.0, 13.0]}
    base.update(cols)
    return pd.DataFrame(base, index=_IDX)


def _signal() -> pd.Series:
    return pd.Series([1, 0, -1, 1], index=_IDX)


def _pred() -> pd.Series:
    return pd.Series([0.9, 0.8, 0.7, 0.6], index=_IDX)


def _values(series: pd.Series) -> list[float]:
    return [float(v) for v in series.tolist()]


class TestAvgVolume:
    def test_lag1_exec_avg_vol_is_mean_up_to_previous_bar(self) -> None:
        df = _df(Volume=[100.0, 200.0, 300.0, 400.0])
        out = build_execution_inputs(df, _signal(), None, "Close", 1)
        # 当日までの平均（min_periods=1）: 100, 150, 200, 250
        assert _values(out.avg_vol) == [100.0, 150.0, 200.0, 250.0]
        # 前の足までの平均。先頭は前の足が無いので 0
        assert _values(out.exec_avg_vol) == [0.0, 100.0, 150.0, 200.0]
        # 先読みを拾えるテストであること: 各足で当日の平均とは違う値
        assert all(a != b for a, b in zip(_values(out.exec_avg_vol), _values(out.avg_vol)))

    def test_lag0_exec_avg_vol_is_same_bar_mean(self) -> None:
        df = _df(Volume=[100.0, 200.0, 300.0, 400.0])
        out = build_execution_inputs(df, _signal(), None, "Close", 0)
        assert _values(out.exec_avg_vol) == [100.0, 150.0, 200.0, 250.0]

    @pytest.mark.parametrize("lag", [0, 1])
    def test_no_volume_column_gives_all_zero(self, lag: int) -> None:
        df = _df()
        out = build_execution_inputs(df, _signal(), None, "Close", lag)
        assert _values(out.avg_vol) == [0.0, 0.0, 0.0, 0.0]
        assert _values(out.exec_avg_vol) == [0.0, 0.0, 0.0, 0.0]


class TestExecColumn:
    def test_lag1_with_open_uses_open(self) -> None:
        df = _df(Open=[9.5, 10.5, 11.5, 12.5])
        out = build_execution_inputs(df, _signal(), None, "Close", 1)
        assert out.exec_col == "Open"

    def test_lag1_without_open_falls_back_to_close_col(self) -> None:
        df = _df().rename(columns={"Close": "Adj Close"})
        out = build_execution_inputs(df, _signal(), None, "Adj Close", 1)
        assert out.exec_col == "Adj Close"

    def test_lag0_uses_close_col_even_with_open(self) -> None:
        df = _df(Open=[9.5, 10.5, 11.5, 12.5])
        out = build_execution_inputs(df, _signal(), None, "Close", 0)
        assert out.exec_col == "Close"


class TestSignalAndPred:
    def test_lag1_shifts_signal_and_pred_by_one_bar(self) -> None:
        df = _df()
        out = build_execution_inputs(df, _signal(), _pred(), "Close", 1)
        sig = out.exec_signal.tolist()
        assert np.isnan(sig[0])
        assert sig[1:] == [1.0, 0.0, -1.0]
        assert out.exec_pred is not None
        pred = out.exec_pred.tolist()
        assert np.isnan(pred[0])
        assert pred[1:] == [0.9, 0.8, 0.7]

    def test_lag2_shifts_signal_by_two_bars(self) -> None:
        df = _df()
        out = build_execution_inputs(df, _signal(), None, "Close", 2)
        sig = out.exec_signal.tolist()
        assert np.isnan(sig[0]) and np.isnan(sig[1])
        assert sig[2:] == [1.0, 0.0]

    def test_lag1_reindexes_signal_to_df_index(self) -> None:
        df = _df()
        # シグナルが一部の足にしか無いときは df の足にそろえてからずらす
        partial = pd.Series([1, -1], index=[_IDX[0], _IDX[2]])
        out = build_execution_inputs(df, partial, None, "Close", 1)
        assert list(out.exec_signal.index) == list(_IDX)
        sig = out.exec_signal.tolist()
        assert np.isnan(sig[0])
        assert sig[1] == 1.0
        assert np.isnan(sig[2])
        assert sig[3] == -1.0

    def test_lag0_keeps_signal_and_pred_as_is(self) -> None:
        df = _df()
        signal = _signal()
        pred = _pred()
        out = build_execution_inputs(df, signal, pred, "Close", 0)
        assert out.exec_signal is signal
        assert out.exec_pred is pred

    @pytest.mark.parametrize("lag", [0, 1])
    def test_pred_none_gives_none(self, lag: int) -> None:
        out = build_execution_inputs(_df(), _signal(), None, "Close", lag)
        assert out.exec_pred is None


class TestAtr:
    def test_lag1_prefers_atr_lag1_as_is(self) -> None:
        df = _df(atr=[1.0, 2.0, 3.0, 4.0], atr_lag1=[0.5, 1.0, 2.0, 3.0])
        out = build_execution_inputs(df, _signal(), None, "Close", 1)
        assert out.exec_atr is not None
        assert _values(out.exec_atr) == [0.5, 1.0, 2.0, 3.0]

    def test_lag1_without_atr_lag1_shifts_atr(self) -> None:
        df = _df(atr=[1.0, 2.0, 3.0, 4.0])
        out = build_execution_inputs(df, _signal(), None, "Close", 1)
        assert out.exec_atr is not None
        atr = out.exec_atr.tolist()
        assert np.isnan(atr[0])
        assert atr[1:] == [1.0, 2.0, 3.0]

    def test_lag0_uses_same_bar_atr(self) -> None:
        df = _df(atr=[1.0, 2.0, 3.0, 4.0], atr_lag1=[0.5, 1.0, 2.0, 3.0])
        out = build_execution_inputs(df, _signal(), None, "Close", 0)
        assert out.exec_atr is not None
        assert _values(out.exec_atr) == [1.0, 2.0, 3.0, 4.0]

    def test_lag0_with_only_atr_lag1_gives_none(self) -> None:
        df = _df(atr_lag1=[0.5, 1.0, 2.0, 3.0])
        out = build_execution_inputs(df, _signal(), None, "Close", 0)
        assert out.exec_atr is None

    @pytest.mark.parametrize("lag", [0, 1])
    def test_no_atr_columns_gives_none(self, lag: int) -> None:
        out = build_execution_inputs(_df(), _signal(), None, "Close", lag)
        assert out.exec_atr is None
