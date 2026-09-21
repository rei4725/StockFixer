"""エントリー株数が整数で、予算を超えないこと。"""

import unittest

from src.backtest.execution import ExecutionModel, TradingCosts


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


if __name__ == "__main__":
    unittest.main()
