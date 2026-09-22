"""domain へ移設した screening 共有型の定義が変わっていないこと。"""

import unittest
from dataclasses import fields

from src.domain.types import HoldRules, PositionEvent, TrendCandidate


class TestHoldRules(unittest.TestCase):
    def test_defaults(self):
        r = HoldRules()
        self.assertEqual(r.trail_ma_weeks, 40)
        self.assertAlmostEqual(r.trail_stop_pct, 0.35)
        self.assertEqual(r.scale_out_multiples, [2.0, 5.0])
        self.assertAlmostEqual(r.scale_out_fraction, 0.20)

    def test_scale_out_multiples_not_shared_between_instances(self):
        """default_factory なので、インスタンス間で list を共有しない。"""
        a, b = HoldRules(), HoldRules()
        a.scale_out_multiples.append(10.0)
        self.assertEqual(b.scale_out_multiples, [2.0, 5.0])


class TestFieldNames(unittest.TestCase):
    def test_position_event_fields(self):
        self.assertEqual(
            [f.name for f in fields(PositionEvent)],
            ["date", "action", "price", "held_fraction", "reason", "multiple"],
        )

    def test_trend_candidate_fields(self):
        self.assertEqual(
            [f.name for f in fields(TrendCandidate)],
            [
                "market",
                "symbol",
                "score",
                "close",
                "dist_from_52w_high",
                "above_200dma",
                "sma200_rising",
                "return_6m",
                "return_12m",
                "avg_volume",
            ],
        )

    def test_trend_candidate_requires_all_fields(self):
        with self.assertRaises(TypeError):
            TrendCandidate(market="us", symbol="AAPL")  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
