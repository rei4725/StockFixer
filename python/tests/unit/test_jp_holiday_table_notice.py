"""東証の休場日の表が切れる前の知らせのテスト（#1663）。

日付はすべて now_jst の差し替えと引数で渡し、今日の日付を読まない。
"""

import unittest
from datetime import date, datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from src.domain import jp_market_holidays
from src.domain.ports import AlertLevel
from src.orchestration.jobs import common

JST = ZoneInfo("Asia/Tokyo")
ADAPTER = "src.infrastructure.discord_notification_adapter.DiscordNotificationAdapter"

BEFORE_START = date(2027, 9, 30)  # 木
START = date(2027, 10, 1)  # 金
NEXT_BUSINESS_DAY = date(2027, 10, 4)  # 月
SATURDAY = date(2027, 10, 2)
HOLIDAY = date(2027, 11, 3)  # 文化の日（水）


def _check(day: date) -> bool:
    now = datetime(day.year, day.month, day.day, 8, 50, tzinfo=JST)
    with patch.object(common, "now_jst", return_value=now):
        return common._skip_if_jp_market_closed("自動発注")


class TestMissingYear(unittest.TestCase):
    def test_answers_the_year_to_add_from_october_of_the_last_year(self):
        from src.domain.jp_market_holidays import missing_jp_holiday_table_year

        self.assertIsNone(missing_jp_holiday_table_year(BEFORE_START))
        self.assertEqual(missing_jp_holiday_table_year(START), 2028)
        self.assertEqual(missing_jp_holiday_table_year(date(2028, 1, 4)), 2028)

    def test_next_year_in_table_is_enough(self):
        from src.domain.jp_market_holidays import missing_jp_holiday_table_year

        with patch.dict(jp_market_holidays._HOLIDAYS_BY_YEAR, {2028: frozenset()}):
            self.assertIsNone(missing_jp_holiday_table_year(START))


class TestNotice(unittest.TestCase):
    def setUp(self):
        patcher = patch(ADAPTER)
        self.adapter = patcher.start()
        self.addCleanup(patcher.stop)
        self.send_alert: MagicMock = self.adapter.return_value.send_alert
        self.send_alert.return_value = True

    def test_day_before_start_is_silent(self):
        self.assertFalse(_check(BEFORE_START))
        self.send_alert.assert_not_called()

    def test_start_day_notifies_the_missing_year(self):
        self.assertFalse(_check(START))
        self.send_alert.assert_called_once()
        title, message = self.send_alert.call_args.args
        self.assertIn("2028", title)
        self.assertIn("jp_market_holidays.py", message)
        self.assertIs(self.send_alert.call_args.kwargs["level"], AlertLevel.WARNING)

    def test_next_year_in_table_is_silent(self):
        with patch.dict(jp_market_holidays._HOLIDAYS_BY_YEAR, {2028: frozenset()}):
            self.assertFalse(_check(START))
        self.send_alert.assert_not_called()

    def test_second_job_of_the_day_does_not_send_again(self):
        _check(START)
        _check(START)
        self.send_alert.assert_called_once()

    def test_next_business_day_sends_again(self):
        _check(START)
        _check(NEXT_BUSINESS_DAY)
        self.assertEqual(self.send_alert.call_count, 2)

    def test_day_is_sent_again_only_after_failed_send(self):
        self.send_alert.return_value = False
        self.assertFalse(_check(START))
        self.send_alert.return_value = True
        self.assertFalse(_check(START))
        self.assertFalse(_check(START))
        self.assertEqual(self.send_alert.call_count, 2)

    def test_closed_days_keep_skipping_without_notice(self):
        self.assertTrue(_check(SATURDAY))
        self.assertTrue(_check(HOLIDAY))
        self.send_alert.assert_not_called()


class TestNoticeFailureDoesNotChangeOrders(unittest.TestCase):
    def test_send_error_keeps_business_day_open(self):
        with patch(ADAPTER) as adapter:
            adapter.return_value.send_alert.side_effect = RuntimeError("discord down")
            with self.assertLogs("src.orchestration.jobs.jp_holiday_table_notice", "ERROR"):
                self.assertFalse(_check(START))
        adapter.return_value.send_alert.assert_called_once()

    def test_marker_write_error_keeps_business_day_open(self):
        with (
            patch(ADAPTER) as adapter,
            patch(
                "src.orchestration.jobs.jp_holiday_table_notice.ensure_dir",
                side_effect=OSError("disk full"),
            ),
            self.assertLogs("src.orchestration.jobs.jp_holiday_table_notice", "ERROR"),
        ):
            self.assertFalse(_check(START))
        adapter.return_value.send_alert.assert_called_once()

    def test_send_error_keeps_holiday_skip(self):
        with patch(ADAPTER) as adapter:
            adapter.return_value.send_alert.side_effect = RuntimeError("discord down")
            self.assertTrue(_check(HOLIDAY))


if __name__ == "__main__":
    unittest.main()
