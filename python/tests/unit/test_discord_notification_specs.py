"""ユニットテスト: discord_notification_specs"""

import unittest

from src.api.discord_notification_specs import (
    COLOR_CAUTION,
    COLOR_INFO,
    COLOR_WARNING,
    DAILY_ORDER_COMPLETION,
    DAILY_ORDER_STOPPED,
    WALK_FORWARD_REPORT_COMPLETION,
    WALK_FORWARD_REPORT_WITH_ERRORS,
    get_daily_order_spec,
    get_optimization_spec,
    get_walk_forward_report_spec,
)


class TestDiscordNotificationSpecs(unittest.TestCase):
    def test_daily_order_spec_switches_on_stop_flag(self):
        self.assertEqual(get_daily_order_spec(trading_stopped=False), DAILY_ORDER_COMPLETION)
        self.assertEqual(get_daily_order_spec(trading_stopped=True), DAILY_ORDER_STOPPED)
        self.assertEqual(DAILY_ORDER_COMPLETION.color, COLOR_INFO)
        self.assertEqual(DAILY_ORDER_STOPPED.color, COLOR_WARNING)

    def test_optimization_spec_uses_caution_when_failures_exist(self):
        self.assertEqual(get_optimization_spec(failed=0).color, 0x00FF00)
        self.assertEqual(get_optimization_spec(failed=1).color, COLOR_CAUTION)

    def test_walk_forward_report_spec_switches_title_and_color(self):
        self.assertEqual(
            get_walk_forward_report_spec(failed_count=0), WALK_FORWARD_REPORT_COMPLETION
        )
        self.assertEqual(
            get_walk_forward_report_spec(failed_count=2), WALK_FORWARD_REPORT_WITH_ERRORS
        )


if __name__ == "__main__":
    unittest.main()
