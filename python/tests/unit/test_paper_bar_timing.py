"""ユニットテスト: ペーパーの約定の日付の部品（bar_timing）を直に確かめる（#1855）

parse_predicted_date はシグナルを作った日を、bar_date_jst は OHLCV の足の日を JST で返す。
paper_broker はこの2つを比べて約定の可否を決めるため、日付の読み違いが紙の成績を黙ってずらす。
"""

import unittest
from datetime import date

import pandas as pd

from src.trading.brokers.paper.bar_timing import bar_date_jst, parse_predicted_date

_LOGGER = "src.trading.brokers.paper.bar_timing"


class TestParsePredictedDate(unittest.TestCase):
    def test_compact_timestamp(self):
        self.assertEqual(parse_predicted_date("20260213_142903"), date(2026, 2, 13))

    def test_iso_timestamp_falls_back(self):
        self.assertEqual(parse_predicted_date("2026-02-13T14:29:03"), date(2026, 2, 13))

    def test_unreadable_returns_none_with_warning(self):
        with self.assertLogs(_LOGGER, level="WARNING") as cm:
            self.assertIsNone(parse_predicted_date("not-a-date"))
        self.assertTrue(any("not-a-date" in line for line in cm.output))

    def test_empty_string_returns_none(self):
        self.assertIsNone(parse_predicted_date(""))

    def test_none_returns_none(self):
        self.assertIsNone(parse_predicted_date(None))


class TestBarDateJst(unittest.TestCase):
    def test_utc_index_crosses_to_next_day_in_jst(self):
        ts = pd.Timestamp("2026-02-12 16:00", tz="UTC")
        self.assertEqual(bar_date_jst(ts), date(2026, 2, 13))

    def test_naive_index_keeps_date(self):
        self.assertEqual(bar_date_jst(pd.Timestamp("2026-02-12 16:00")), date(2026, 2, 12))

    def test_naive_date_string_keeps_date(self):
        self.assertEqual(bar_date_jst("2026-02-12"), date(2026, 2, 12))


if __name__ == "__main__":
    unittest.main()
