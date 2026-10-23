"""
Unit Test: バックテストの3つの道が当日の atr を特徴量に入れないこと (#994)

load_features の api・raw は当日の高安引けで作った atr を、file は atr_lag1 の写しを
atr として表に載せる。本番の特徴量はラグの列だけなので、run_backtest_single・
WalkForwardValidator・_build_signal_matrix の学習と予測に渡る列から、atr を
Close・close・Open・Volume と同じく外す。表の atr は ATR サイジング用に残す。
"""

from typing import Any, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.backtest.data_port import set_backtest_data_port
from src.market_data.technical import create_basic_lag_features

_N = 80
_FORBIDDEN = ("atr", "Close", "close", "Open", "Volume")


def _raw_ohlcv() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=_N, freq="B", name="Date")
    rng = np.random.default_rng(0)
    close = 100.0 + rng.standard_normal(_N).cumsum()
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0 + rng.random(_N),
            "Low": close - 1.0 - rng.random(_N),
            "Close": close,
            "Volume": 10_000.0 + np.arange(_N, dtype=float) * 100.0,
        },
        index=dates,
    )


def _add_atr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr"] = df["High"] - df["Low"]
    return df


class _FakeDataPort:
    """生の OHLCV を返し、指標は当日の高安で作る atr だけ足す偽のデータポート。"""

    def __init__(self, raw: Optional[pd.DataFrame]) -> None:
        self.raw = raw

    def get_stock_data(self, market: str, ticker: str, start: Any, end: Any) -> Any:
        return None if self.raw is None else self.raw.copy()

    def get_raw_ohlcv_from_db(
        self, market: str, symbol: str, start: Any = None, end: Any = None
    ) -> Any:
        return None if self.raw is None else self.raw.copy()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return _add_atr(df)

    def create_basic_lag_features(self, df: pd.DataFrame, n_lags: int = 10) -> Any:
        return create_basic_lag_features(df, n_lags=n_lags)


def _stock_features_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """stock_features の形（ラグの列と date と y。atr_lag1 を含む）を作る。"""
    X, y = create_basic_lag_features(_add_atr(raw), n_lags=10)
    out = X.copy()
    out["y"] = y
    out.insert(0, "date", X.index.strftime("%Y-%m-%d"))
    return out.reset_index(drop=True)


@pytest.fixture(params=["api", "raw", "file"])
def features(request, monkeypatch) -> pd.DataFrame:
    """各 source の load_features の出力"""
    raw = _raw_ohlcv()
    set_backtest_data_port(_FakeDataPort(raw))
    monkeypatch.setattr("src.utils.db.load_stock_features", lambda m, s: _stock_features_frame(raw))

    from src.backtest.pipeline import load_features

    df = load_features("jp", "7203", request.param)
    assert "atr" in df.columns
    assert "atr_lag1" in df.columns
    return df


def _assert_no_same_day_columns(cols: list[str]) -> None:
    for col in _FORBIDDEN:
        assert col not in cols
    assert "atr_lag1" in cols
    assert "Close_lag1" in cols


class TestRunBacktestSingleExcludesAtr:
    @pytest.mark.parametrize("ensemble", [False, True])
    def test_train_and_predict_exclude_atr(self, features, ensemble):
        n_test = len(features) - int(len(features) * 0.8)
        with (
            patch("src.backtest.pipeline.runner.load_features", return_value=features),
            patch("src.backtest.pipeline.runner.get_model_manager") as mock_get_mm,
            patch("src.backtest.pipeline.runner._ensemble_predict") as mock_ens,
            patch("src.backtest.backtester.Backtester") as mock_bt,
        ):
            mock_mm = mock_get_mm.return_value
            mock_mm.predict_with_model.side_effect = lambda name, X: np.zeros(len(X))
            mock_ens.side_effect = lambda mm, X_tr, y_tr, X_te, name: pd.Series(
                np.zeros(len(X_te)), index=X_te.index
            )
            mock_bt.return_value.simulate_trading.return_value = (pd.DataFrame(), {})

            from src.backtest.pipeline import run_backtest_single

            run_backtest_single("jp", "7203", ensemble=ensemble, position_sizing="atr")

        if ensemble:
            _, X_train, _, X_test, _ = mock_ens.call_args.args
        else:
            X_train = mock_mm.train_model.call_args.args[1]
            X_test = mock_mm.predict_with_model.call_args.args[1]
        _assert_no_same_day_columns(list(X_train.columns))
        _assert_no_same_day_columns(list(X_test.columns))
        assert len(X_test) == n_test

        # ATR サイジング用の atr・atr_lag1 は simulate_trading に渡す表に元の値で残る
        sim_df = mock_bt.return_value.simulate_trading.call_args.args[0]
        pd.testing.assert_series_equal(sim_df["atr"], features.loc[sim_df.index, "atr"])
        pd.testing.assert_series_equal(sim_df["atr_lag1"], features.loc[sim_df.index, "atr_lag1"])


class TestWalkForwardExcludesAtr:
    @pytest.mark.parametrize("ensemble", [False, True])
    def test_train_and_predict_exclude_atr(self, features, ensemble):
        from src.backtest.walk_forward import WalkForwardValidator

        model_manager = MagicMock()
        model = MagicMock()
        model.predict.side_effect = lambda X: np.zeros(len(X))
        model_manager.get_model.return_value = model
        v = WalkForwardValidator(
            market="jp",
            symbol="7203",
            model_manager=model_manager,
            signal_generator=MagicMock(),
            n_splits=2,
            ensemble=ensemble,
            position_sizing="atr",
        )
        with (
            patch.object(v, "_load_features", return_value=features.copy()),
            patch(
                "src.backtest.pipeline._ensemble_predict",
                side_effect=lambda mm, X_tr, y_tr, X_v, name: pd.Series(
                    np.zeros(len(X_v)), index=X_v.index
                ),
            ) as mock_ens,
            patch("src.backtest.backtester.Backtester") as mock_bt,
        ):
            mock_bt.return_value.simulate_trading.return_value = (
                pd.DataFrame(),
                {"total_return": 0.0},
            )
            result = v.run(model_name="StockXGBoostModel")

        assert "error" not in result.columns
        calls = mock_ens.call_args_list if ensemble else model.train.call_args_list
        assert len(calls) == 2
        if ensemble:
            frames = [c.args[1] for c in calls] + [c.args[3] for c in calls]
        else:
            frames = [c.args[0] for c in calls] + [c.args[0] for c in model.predict.call_args_list]
        for X in frames:
            _assert_no_same_day_columns(list(X.columns))

        sim_df = mock_bt.return_value.simulate_trading.call_args.args[0]
        pd.testing.assert_series_equal(sim_df["atr"], features.loc[sim_df.index, "atr"])
        pd.testing.assert_series_equal(sim_df["atr_lag1"], features.loc[sim_df.index, "atr_lag1"])


class TestSignalMatrixExcludesAtr:
    @pytest.mark.parametrize("ensemble", [False, True])
    def test_train_and_predict_exclude_atr(self, features, ensemble):
        from src.backtest.portfolio import _build_signal_matrix

        with (
            patch("src.backtest.pipeline.load_features", return_value=features),
            patch("src.backtest.portfolio.signal_matrix.get_model_manager") as mock_get_mm,
            patch(
                "src.backtest.pipeline._ensemble_predict",
                side_effect=lambda mm, X_tr, y_tr, X_te, name: pd.Series(
                    np.zeros(len(X_te)), index=X_te.index
                ),
            ) as mock_ens,
        ):
            mock_mm = mock_get_mm.return_value
            mock_mm.predict_with_model.side_effect = lambda name, X: np.zeros(len(X))
            score_matrix, _ = _build_signal_matrix(
                symbols=[("jp", "7203")],
                model_type="XGBoostModel",
                train_ratio=0.8,
                source="file",
                threshold=0.0,
                ensemble=ensemble,
            )

        assert not score_matrix.empty
        if ensemble:
            _, X_train, _, X_test, _ = mock_ens.call_args.args
        else:
            X_train = mock_mm.train_model.call_args.args[1]
            X_test = mock_mm.predict_with_model.call_args.args[1]
        _assert_no_same_day_columns(list(X_train.columns))
        _assert_no_same_day_columns(list(X_test.columns))


class TestNonFeatureColumnsDefinedOnce:
    def test_contains_same_day_columns(self):
        from src.backtest.pipeline.features import NON_FEATURE_COLUMNS

        assert set(_FORBIDDEN) <= set(NON_FEATURE_COLUMNS)
        assert "atr_lag1" not in NON_FEATURE_COLUMNS
