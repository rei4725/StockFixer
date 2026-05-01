"""ユニットテスト: japan_time"""

import unittest
from datetime import datetime, timezone

from src.utils.japan_time import (
    format_jst,
    format_jst_from_iso,
    isoformat_jst,
    isoformat_utc,
    now_utc,
    parse_iso_datetime,
    to_jst,
    to_utc,
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

    def test_now_utc_returns_utc_datetime(self):
        result = now_utc()
        self.assertEqual(result.tzinfo, timezone.utc)

    def test_to_utc_naive_datetime_assigns_utc(self):
        naive = datetime(2026, 4, 6, 12, 0, 0)
        result = to_utc(naive)
        self.assertEqual(result.tzinfo, timezone.utc)
        self.assertEqual(result.hour, 12)

    def test_to_utc_aware_datetime_converts(self):
        from zoneinfo import ZoneInfo

        jst = ZoneInfo("Asia/Tokyo")
        source = datetime(2026, 4, 6, 21, 0, 0, tzinfo=jst)
        result = to_utc(source)
        self.assertEqual(result.tzinfo, timezone.utc)
        self.assertEqual(result.hour, 12)

    def test_parse_iso_datetime_naive_string_assigns_utc(self):
        result = parse_iso_datetime("2026-04-06T12:00:00")
        self.assertEqual(result.tzinfo, timezone.utc)

    def test_parse_iso_datetime_z_suffix(self):
        result = parse_iso_datetime("2026-04-06T12:00:00Z")
        self.assertEqual(result.tzinfo.utcoffset(result).total_seconds(), 0)

    def test_isoformat_utc_no_arg_returns_current_utc(self):
        result = isoformat_utc()
        self.assertIn("+00:00", result)

    def test_isoformat_jst_no_arg_returns_jst_offset(self):
        result = isoformat_jst()
        self.assertIn("+09:00", result)

    def test_format_jst_from_iso_empty_string_returns_fallback(self):
        result = format_jst_from_iso("")
        self.assertEqual(result, "不明")

    def test_format_jst_from_iso_none_returns_fallback(self):
        result = format_jst_from_iso(None)
        self.assertEqual(result, "不明")

    def test_format_jst_from_iso_invalid_returns_original(self):
        result = format_jst_from_iso("not-a-date")
        self.assertEqual(result, "not-a-date")

    def test_format_jst_from_iso_custom_fallback(self):
        result = format_jst_from_iso(None, fallback="N/A")
        self.assertEqual(result, "N/A")


if __name__ == "__main__":
    unittest.main()
