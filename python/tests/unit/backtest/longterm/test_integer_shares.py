"""エントリー株数が整数で、予算を超えないこと。"""

import unittest
from unittest.mock import patch

import pandas as pd

from src.backtest.execution import ExecutionModel, TradingCosts
from src.backtest.longterm import engine
from src.backtest.longterm.config import LongtermBacktestConfig
from src.backtest.longterm.portfolio import Portfolio
from src.screening.types import PositionEvent, TrendCandidate


class TestIntegerShares(unittest.TestCase):
    def test_qty_is_integer_and_within_budget(self):
        ex = ExecutionModel(TradingCosts(fee_rate=0.001, slippage_rate=0.0005))
        budget, price = 100_000.0, 333.33
        qty = ex.max_affordable_qty(budget, price)
        self.assertIsInstance(qty, int)
        self.assertLessEqual(ex.buy_cost(qty, price), budget)
        self.assertGreater(ex.buy_cost(qty + 1, price), budget)

    def test_expensive_symbol_yields_zero(self):
        ex = ExecutionModel(TradingCosts())
        self.assertEqual(ex.max_affordable_qty(100.0, 1_000.0), 0)


class TestEnterCandidatesIntegerSizing(unittest.TestCase):
    """`enter_candidates` の配線レベルで、端株切り捨て＋余剰現金温存を検証する。

    `ExecutionModel.max_affordable_qty` / `buy_cost` を単体で叩くだけでは、
    `enter_candidates` が実際にそれらを正しく呼び出し、`cost_basis` に
    実コストを、`portfolio.cash` に端株分の余剰を残しているかは検証できない
    （そこは `Portfolio.enter` が `cash -= cost_basis` を機械的に行うだけなので、
    `cost_basis` に何を渡しても cash 非負性等の既存不変条件は自動的に満たされて
    しまう）。このテストは `per_position / entry_price` が割り切れない金額・価格を
    選び、エントリー後の `shares` / `cost_basis` / `portfolio.cash` を直接検証する。
    """

    _DATE = "2024-01-02"
    _ENTRY_PRICE = 333.33
    _PER_POSITION = 1000.0  # max_positions=1 なので empty_slots=1 → per_position = cash

    def _run_enter_candidates(self):
        price_map = {
            "X": pd.DataFrame({"date": [self._DATE], "Close": [self._ENTRY_PRICE]}),
        }
        candidate = TrendCandidate(
            market="us",
            symbol="X",
            score=1.0,
            close=self._ENTRY_PRICE,
            dist_from_52w_high=0.0,
            above_200dma=True,
            sma200_rising=True,
            return_6m=0.1,
            return_12m=0.2,
            avg_volume=1_000_000.0,
        )
        ev = PositionEvent(
            date=self._DATE,
            action="exit",
            price=self._ENTRY_PRICE,
            held_fraction=0.0,
            reason="x",
            multiple=1.0,
        )

        cfg = LongtermBacktestConfig.build(market="us", max_positions=1)
        execution = ExecutionModel(cfg.costs)
        portfolio = Portfolio(cash=self._PER_POSITION)

        with patch.object(
            engine, "screen_trend_candidates", return_value=[candidate]
        ), patch.object(engine, "simulate_position", return_value=[ev]):
            engine.enter_candidates(cfg, portfolio, self._DATE, price_map, execution)

        return portfolio, execution

    def test_uneven_division_yields_int_shares_and_leftover_cash(self):
        portfolio, execution = self._run_enter_candidates()

        self.assertIn("X", portfolio.positions)
        pos = portfolio.positions["X"]

        # 端株切り捨て: 整数かつ予算を丸ごと使い切る株数（3.0000...）未満。
        self.assertIsInstance(pos.shares, int)
        self.assertLess(pos.shares, self._PER_POSITION / self._ENTRY_PRICE)

        # cost_basis は実際の buy_cost（= per_position ではない）。
        expected_cost = execution.buy_cost(pos.shares, self._ENTRY_PRICE)
        self.assertEqual(pos.cost_basis, expected_cost)
        self.assertLess(pos.cost_basis, self._PER_POSITION)

        # 余剰 (per_position - cost_basis) は現金として残る。
        self.assertEqual(portfolio.cash, self._PER_POSITION - pos.cost_basis)


if __name__ == "__main__":
    unittest.main()
