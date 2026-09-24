"""metrics/stats.py の純粋統計関数。"""

import math
import unittest

import pandas as pd

from src.backtest.metrics import stats


class TestMaxDrawdown(unittest.TestCase):
    def test_monotonic_rise_has_no_drawdown(self):
        self.assertAlmostEqual(stats.max_drawdown(pd.Series([1.0, 2.0, 3.0])), 0.0)

    def test_halving_is_minus_half(self):
        self.assertAlmostEqual(stats.max_drawdown(pd.Series([100.0, 50.0])), -0.5)

    def test_empty_series_is_zero(self):
        self.assertAlmostEqual(stats.max_drawdown(pd.Series(dtype=float)), 0.0)


class TestSharpe(unittest.TestCase):
    def test_zero_variance_is_zero(self):
        self.assertAlmostEqual(stats.sharpe_per_trade([1.0, 1.0, 1.0]), 0.0)

    def test_annualize_scales_by_sqrt(self):
        self.assertAlmostEqual(stats.annualize_sharpe(0.5, 4.0), 0.5 * math.sqrt(4.0))

    def test_annualize_zero_frequency_is_zero(self):
        self.assertAlmostEqual(stats.annualize_sharpe(0.5, 0.0), 0.0)


class TestProfitFactor(unittest.TestCase):
    def test_ratio_of_gross_win_to_gross_loss(self):
        self.assertAlmostEqual(stats.profit_factor([10.0, 20.0], [-10.0]), 3.0)

    def test_no_losses_is_infinite(self):
        self.assertEqual(stats.profit_factor([10.0], []), math.inf)

    def test_no_trades_is_zero(self):
        self.assertAlmostEqual(stats.profit_factor([], []), 0.0)


class TestCagr(unittest.TestCase):
    def test_doubling_in_one_year(self):
        self.assertAlmostEqual(stats.cagr(100.0, 200.0, 1.0), 1.0)

    def test_total_loss_is_minus_one(self):
        self.assertAlmostEqual(stats.cagr(100.0, 0.0, 1.0), -1.0)


if __name__ == "__main__":
    unittest.main()
