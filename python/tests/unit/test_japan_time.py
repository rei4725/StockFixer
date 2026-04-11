"""ユニットテスト: japan_time"""

import unittest
from datetime import datetime, timezone

from src.utils.japan_time import (
    format_jst,
    format_jst_from_iso,
    isoformat_jst,
    isoformat_utc,
    to_jst,
)


class TestJapanTime(unittest.TestCase):
    def test_to_jst_converts_from_utc(self):
        source = datetime(2026, 4, 6, 0, 30, 45, tzinfo=timezone.utc)

        converted = to_jst(source)

        self.assertEqual(converted.strftime("%Y-%m-%d %H:%M:%S %Z"), "2026-04-06 09:30:45 JST")

    def test_format_jst_applies_timezone_to_naive_datetime(self):
        source = datetime(2026, 4, 6, 9, 30, 45)

        formatted = format_jst(source, fmt="%Y-%m-%d %H:%M:%S %Z")

        self.assertEqual(formatted, "2026-04-06 09:30:45 JST")

    def test_isoformat_jst_returns_offset(self):
        source = datetime(2026, 4, 6, 0, 30, 45, tzinfo=timezone.utc)

        formatted = isoformat_jst(source)

        self.assertEqual(formatted, "2026-04-06T09:30:45+09:00")

    def test_isoformat_utc_returns_utc_offset(self):
        source = datetime(2026, 4, 6, 9, 30, 45, tzinfo=timezone.utc)

        formatted = isoformat_utc(source)

        self.assertEqual(formatted, "2026-04-06T09:30:45+00:00")

    def test_format_jst_from_iso_converts_utc_string_for_display(self):
        formatted = format_jst_from_iso("2026-04-06T00:30:45+00:00")

        self.assertEqual(formatted, "2026-04-06 09:30:45 JST")


if __name__ == "__main__":
    unittest.main()
