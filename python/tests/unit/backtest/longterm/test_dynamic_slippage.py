"""長期バックテストの動的スリッページ（市場インパクト）のテスト（#1310）。"""

import unittest
from unittest.mock import patch

import pandas as pd

from src.backtest.execution import ExecutionModel
from src.backtest.longterm import engine
from src.backtest.longterm import prices as lb_prices
from src.backtest.longterm.config import LongtermBacktestConfig
from src.backtest.longterm.portfolio import Portfolio
from src.backtest.screening_port import set_backtest_screening_port
from src.backtest.slippage import make_slippage_fn
from src.domain.types import PositionEvent, TrendCandidate

_N = 25
_DATES = pd.date_range("2024-01-01", periods=_N, freq="B")
_ENTRY_IDX = 22
_ENTRY_DATE = _DATES[_ENTRY_IDX].strftime("%Y-%m-%d")
_PRICE = 100.0
_THIN_VOLUME = 1_000.0
_CASH = 50_000.0


class _StubScreeningPort:
    """simulate_position だけを固定イベントで返すスタブ。"""

    def screen_trend_candidates(self, market, top_n, as_of):
        return []

    def simulate_position(self, prices, entry_date, rules):
        return [
            PositionEvent(
                date=entry_date,
                action="entry",
                price=_PRICE,
                held_fraction=1.0,
                reason="entry",
                multiple=1.0,
            )
        ]


def _raw(entry_volume: float = _THIN_VOLUME) -> pd.DataFrame:
    volumes = [_THIN_VOLUME] * _N
    volumes[_ENTRY_IDX] = entry_volume
    return pd.DataFrame(
        {
            "symbol": "THIN",
            "ts": _DATES,
            "close": [_PRICE] * _N,
            "volume": volumes,
        }
    )


def _candidate() -> TrendCandidate:
    return TrendCandidate(
        market="us",
        symbol="THIN",
        score=1.0,
        close=_PRICE,
        dist_from_52w_high=0.0,
        above_200dma=True,
        sma200_rising=True,
        return_6m=0.1,
        return_12m=0.2,
        avg_volume=_THIN_VOLUME,
    )


def _enter(execution: ExecutionModel, entry_volume: float = _THIN_VOLUME) -> Portfolio:
    with patch.object(lb_prices, "load_raw_closes", return_value=_raw(entry_volume)):
        price_map = lb_prices.load_price_map("us", _ENTRY_DATE)
    cfg = LongtermBacktestConfig.build(market="us", max_positions=1)
    portfolio = Portfolio(cash=_CASH)
    set_backtest_screening_port(_StubScreeningPort())
    engine.enter_candidates(cfg, portfolio, _ENTRY_DATE, [_candidate()], price_map, execution)
    return portfolio


class TestLongtermDynamicSlippage(unittest.TestCase):
    def setUp(self):
        self.costs = LongtermBacktestConfig.build(market="us").costs

    def test_thin_symbol_buy_pays_market_impact(self):
        """平均出来高の小さい銘柄を買うと、現金の減りが上乗せの分だけ多い。"""
        dynamic = ExecutionModel(self.costs, make_slippage_fn())
        flat = ExecutionModel(self.costs)
        portfolio = _enter(dynamic)

        pos = portfolio.positions["THIN"]
        expected = dynamic.buy_cost(pos.shares, _PRICE, _THIN_VOLUME)
        self.assertAlmostEqual(pos.cost_basis, expected)
        self.assertGreater(pos.cost_basis, flat.buy_cost(pos.shares, _PRICE))
        self.assertAlmostEqual(portfolio.cash, _CASH - expected)

    def test_entry_day_volume_is_not_used(self):
        """約定の日の出来高を膨らませても、その日のコストは変わらない（先読みしない）。"""
        base = _enter(ExecutionModel(self.costs, make_slippage_fn()))
        inflated = _enter(ExecutionModel(self.costs, make_slippage_fn()), entry_volume=1e12)
        self.assertEqual(base.positions["THIN"].shares, inflated.positions["THIN"].shares)
        self.assertEqual(base.positions["THIN"].cost_basis, inflated.positions["THIN"].cost_basis)

    def test_avg_volume_excludes_execution_day(self):
        """約定の日の平均出来高は前の 20 本の平均で、先頭の日は 0（フラットのみ）。"""
        with patch.object(lb_prices, "load_raw_closes", return_value=_raw(1e12)):
            df = lb_prices.load_price_map("us", _ENTRY_DATE)["THIN"]
        self.assertEqual(df["AvgVolume"].iloc[0], 0.0)
        self.assertAlmostEqual(df["AvgVolume"].iloc[_ENTRY_IDX], _THIN_VOLUME)
        self.assertEqual(lb_prices.exec_avg_volume({"THIN": df}, "THIN", _ENTRY_DATE), _THIN_VOLUME)
        self.assertEqual(lb_prices.exec_avg_volume({"THIN": df}, "THIN", "1999-01-01"), 0.0)

    def test_dynamic_slippage_on_by_default_and_can_be_turned_off(self):
        on = engine.build_execution(LongtermBacktestConfig.build(market="us"))
        off = engine.build_execution(
            LongtermBacktestConfig.build(market="us", dynamic_slippage=False)
        )
        self.assertIsNotNone(on.slippage_fn)
        self.assertIsNone(off.slippage_fn)

    def test_off_matches_flat_cost(self):
        """切ったときは出来高があっても今までと同じフラットのコスト。"""
        off = engine.build_execution(
            LongtermBacktestConfig.build(market="us", dynamic_slippage=False)
        )
        pos = _enter(off).positions["THIN"]
        self.assertEqual(pos.cost_basis, ExecutionModel(self.costs).buy_cost(pos.shares, _PRICE))

    def test_sell_passes_avg_volume(self):
        dynamic = ExecutionModel(self.costs, make_slippage_fn())
        portfolio = _enter(dynamic)
        cash_before = portfolio.cash
        ev = PositionEvent(
            date=_ENTRY_DATE,
            action="exit",
            price=_PRICE,
            held_fraction=0.0,
            reason="x",
            multiple=1.0,
        )
        shares = portfolio.positions["THIN"].shares
        portfolio.close("THIN", ev, dynamic, 1.0, volume=_THIN_VOLUME)
        self.assertAlmostEqual(
            portfolio.cash - cash_before, dynamic.sell_proceeds(shares, _PRICE, _THIN_VOLUME)
        )


if __name__ == "__main__":
    unittest.main()
