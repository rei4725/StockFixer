"""結論文が設定を反映すること。"""

import unittest

from src.backtest.longterm.config import LongtermBacktestConfig
from src.backtest.longterm.reporting import build_conclusion
from src.domain.types import HoldRules

_METRICS = {
    "initial_cash": 1_000_000.0,
    "final_cash": 2_000_000.0,
    "total_return": 1.0,
    "cagr": 0.2,
    "max_drawdown": -0.3,
    "n_2x": 1,
    "n_3x": 0,
    "n_5x": 0,
    "n_10x": 0,
}


class TestBuildConclusion(unittest.TestCase):
    def test_weekly_freq_is_reflected(self):
        cfg = LongtermBacktestConfig.build(market="us", rescreen_freq="weekly")
        self.assertIn("週次スクリーン", build_conclusion(_METRICS, cfg))
        self.assertNotIn("四半期", build_conclusion(_METRICS, cfg))

    def test_quarterly_freq_is_reflected(self):
        cfg = LongtermBacktestConfig.build(market="us", rescreen_freq="quarterly")
        self.assertIn("四半期スクリーン", build_conclusion(_METRICS, cfg))

    def test_trail_ma_weeks_is_reflected(self):
        cfg = LongtermBacktestConfig.build(market="us", rules=HoldRules(trail_ma_weeks=30))
        self.assertIn("30週線割れ", build_conclusion(_METRICS, cfg))

    def test_missing_benchmark_is_stated(self):
        cfg = LongtermBacktestConfig.build(market="us")
        self.assertIn("ベンチマーク比較は", build_conclusion(_METRICS, cfg))


if __name__ == "__main__":
    unittest.main()
