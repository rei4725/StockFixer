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

    def test_jp_benchmark_default_is_jp_index(self):
        """jp で benchmark_ticker を渡さないと jp の指数を使う（#1077）。"""
        cfg = LongtermBacktestConfig.build(market="jp")
        self.assertEqual(cfg.benchmark_ticker, "1306.T")

    def test_jp_benchmark_default_without_build(self):
        cfg = LongtermBacktestConfig(market="jp")
        self.assertEqual(cfg.benchmark_ticker, "1306.T")

    def test_explicit_benchmark_wins(self):
        cfg = LongtermBacktestConfig.build(market="jp", benchmark_ticker="^N225")
        self.assertEqual(cfg.benchmark_ticker, "^N225")
        cfg = LongtermBacktestConfig.build(market="us", benchmark_ticker="^IXIC")
        self.assertEqual(cfg.benchmark_ticker, "^IXIC")

    def test_none_benchmark_falls_back_to_market_default(self):
        cfg = LongtermBacktestConfig.build(market="jp", benchmark_ticker=None)
        self.assertEqual(cfg.benchmark_ticker, "1306.T")

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
