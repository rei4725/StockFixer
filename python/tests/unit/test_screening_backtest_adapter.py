"""BacktestScreeningAdapter が Protocol を満たし実関数へ委譲すること。"""

import unittest
from unittest.mock import patch

import pandas as pd

from src.backtest.screening_port import BacktestScreeningPort
from src.domain.types import HoldRules
from src.screening.backtest_adapter import BacktestScreeningAdapter


class TestAdapter(unittest.TestCase):
    def test_satisfies_protocol(self):
        self.assertIsInstance(BacktestScreeningAdapter(), BacktestScreeningPort)

    def test_screen_delegates_with_same_arguments(self):
        with patch("src.screening.trend_screener.screen_trend_candidates") as m:
            m.return_value = ["sentinel"]
            out = BacktestScreeningAdapter().screen_trend_candidates("us", 30, "2024-01-02")
        self.assertEqual(out, ["sentinel"])
        m.assert_called_once_with(market="us", top_n=30, as_of="2024-01-02")

    def test_simulate_delegates_with_same_arguments(self):
        prices = pd.DataFrame({"date": ["2024-01-02"], "Close": [10.0]})
        rules = HoldRules()
        with patch("src.screening.hold_engine.simulate_position") as m:
            m.return_value = ["event"]
            out = BacktestScreeningAdapter().simulate_position(prices, "2024-01-02", rules)
        self.assertEqual(out, ["event"])
        args, kwargs = m.call_args
        self.assertIs(args[0], prices)
        self.assertEqual(kwargs["entry_date"], "2024-01-02")
        self.assertIs(kwargs["rules"], rules)


if __name__ == "__main__":
    unittest.main()
