"""LongtermBacktestConfig の既定値とコスト解決。"""

import unittest
from dataclasses import FrozenInstanceError

from src.backtest.longterm.config import LongtermBacktestConfig


class TestBuild(unittest.TestCase):
    def test_us_slippage_default(self):
        cfg = LongtermBacktestConfig.build(market="us")
        self.assertAlmostEqual(cfg.costs.slippage_rate, 0.0005)
        self.assertAlmostEqual(cfg.costs.fee_rate, 0.001)

    def test_jp_slippage_default(self):
        cfg = LongtermBacktestConfig.build(market="jp")
        self.assertAlmostEqual(cfg.costs.slippage_rate, 0.0010)

    def test_explicit_slippage_wins(self):
        cfg = LongtermBacktestConfig.build(market="us", slippage=0.01)
        self.assertAlmostEqual(cfg.costs.slippage_rate, 0.01)

    def test_defaults(self):
        cfg = LongtermBacktestConfig.build(market="us")
        self.assertEqual(cfg.rescreen_freq, "quarterly")
        self.assertEqual(cfg.top_n, 30)
        self.assertEqual(cfg.max_positions, 10)
        self.assertEqual(cfg.n_trials, 0)
        self.assertEqual(cfg.benchmark_ticker, "^GSPC")

    def test_execution_lag_defaults_to_zero_in_pr4(self):
        """PR-4 は振る舞い不変。lag の既定を 1 にするのは PR-5。"""
        cfg = LongtermBacktestConfig.build(market="us")
        self.assertEqual(cfg.execution_lag, 0)

    def test_is_frozen(self):
        cfg = LongtermBacktestConfig.build(market="us")
        with self.assertRaises(FrozenInstanceError):
            cfg.top_n = 99  # type: ignore[misc]

    def test_passthrough_kwargs(self):
        cfg = LongtermBacktestConfig.build(market="us", top_n=5, max_positions=2)
        self.assertEqual(cfg.top_n, 5)
        self.assertEqual(cfg.max_positions, 2)


if __name__ == "__main__":
    unittest.main()
