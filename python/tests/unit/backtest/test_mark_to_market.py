"""mark_to_market と build_equity_curves を直接確かめる（#1919）。

期待値は手で計算した値で書く。slippage_fn は 10 / 平均出来高 を返すので、
平均出来高 1000 なら 1%、500 なら 2% の上乗せになる。
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.execution import ExecutionModel, TradingCosts
from src.backtest.mark_to_market import build_equity_curves, mark_to_market


def _slippage(qty: int, price: float, avg_volume: float) -> float:
    return 10.0 / avg_volume


def _model() -> ExecutionModel:
    return ExecutionModel(TradingCosts(fee_rate=0.001, slippage_rate=0.0), slippage_fn=_slippage)


class TestMarkToMarket:
    def test_no_position_returns_cash(self) -> None:
        date, net, gross = mark_to_market(
            _model(), "d1", 1000.0, 1200.0, 0, 90.0, 0, 0.0, 1000.0, 500.0
        )
        assert date == "d1"
        assert net == pytest.approx(1000.0)
        assert gross == pytest.approx(1200.0)

    def test_long_only_adds_market_value_without_cost(self) -> None:
        # long_value = 5 * 90 = 450
        _, net, gross = mark_to_market(_model(), "d1", 500.0, 600.0, 5, 90.0, 0, 0.0, 1000.0, 500.0)
        assert net == pytest.approx(950.0)
        assert gross == pytest.approx(1050.0)

    def test_short_only_uses_entry_and_exit_volumes_separately(self) -> None:
        # 建てた時の売り: 10 * 100 * (1 - 0.001 - 10/1000) = 989.0
        # 返す時の買い:   10 * 90 * (1 + 0.001 + 10/500)   = 918.9
        # short_net = 70.1、short_gross = (100 - 90) * 10 = 100
        _, net, gross = mark_to_market(
            _model(), "d1", 1000.0, 1000.0, 0, 90.0, 10, 100.0, 1000.0, 500.0
        )
        assert net == pytest.approx(1070.1)
        assert gross == pytest.approx(1100.0)

    def test_short_swapped_volumes_change_net(self) -> None:
        # 出来高を入れ替えると: 10*100*(1-0.001-0.02)=979.0、10*90*(1+0.001+0.01)=909.9
        # short_net = 69.1。建てた時と返す時の出来高を取り違えると値が変わる
        _, net, _ = mark_to_market(
            _model(), "d1", 1000.0, 1000.0, 0, 90.0, 10, 100.0, 500.0, 1000.0
        )
        assert net == pytest.approx(1069.1)


class TestBuildEquityCurves:
    def test_empty_records_return_empty_series(self) -> None:
        net, gross = build_equity_curves([])
        assert isinstance(net, pd.Series) and net.empty
        assert isinstance(gross, pd.Series) and gross.empty
        assert net.dtype == float and gross.dtype == float

    def test_three_records(self) -> None:
        dates = list(pd.date_range("2026-01-05", periods=3, freq="D"))
        records = [
            (dates[0], 1000.0, 1010.0),
            (dates[1], 990.0, 1005.0),
            (dates[2], 1020.0, 1040.0),
        ]
        net, gross = build_equity_curves(records)
        assert list(net.index) == dates
        assert list(gross.index) == dates
        assert net.tolist() == [1000.0, 990.0, 1020.0]
        assert gross.tolist() == [1010.0, 1005.0, 1040.0]
