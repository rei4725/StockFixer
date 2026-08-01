"""ユニットテスト: alert_service の追加カバレッジ（streak / send）"""

from __future__ import annotations

import unittest
from unittest.mock import patch


class TestGetSetStreak(unittest.TestCase):
    def test_get_streak_returns_value(self):
        from src.utils.alert_service import _get_streak

        with patch("src.utils.db.system_config.get_config_value", return_value="3"):
            result = _get_streak("some_key")
        self.assertEqual(result, 3)

    def test_get_streak_returns_zero_on_none(self):
        from src.utils.alert_service import _get_streak

        with patch("src.utils.db.system_config.get_config_value", return_value=None):
            result = _get_streak("some_key")
        self.assertEqual(result, 0)

    def test_get_streak_returns_zero_on_exception(self):
        from src.utils.alert_service import _get_streak

        with patch(
            "src.utils.db.system_config.get_config_value",
            side_effect=RuntimeError("db error"),
        ):
            result = _get_streak("some_key")
        self.assertEqual(result, 0)

    def test_set_streak_calls_set_config(self):
        from src.utils.alert_service import _set_streak

        with patch("src.utils.db.system_config.set_config_value") as mock_set:
            _set_streak("some_key", 5)
        mock_set.assert_called_once_with("some_key", "5")

    def test_set_streak_handles_exception(self):
        from src.utils.alert_service import _set_streak

        with patch(
            "src.utils.db.system_config.set_config_value",
            side_effect=RuntimeError("db error"),
        ):
            _set_streak("some_key", 5)  # 例外が伝播しなければOK


class TestSendAlertDetail(unittest.TestCase):
    def _make_triggered_result(self):
        from src.utils.alert_service import AlertResult

        r = AlertResult(
            rule_id="R1",
            name="test rule",
            triggered=True,
            consecutive_count=3,
            threshold=3,
            details={},
        )
        return r

    def test_send_alert_detail_calls_webhook(self):
        from src.utils.alert_service import _send_alert_detail

        results = [self._make_triggered_result()]
        notifier = lambda title, msg, color: True  # noqa: E731
        result = _send_alert_detail(results, notifier)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
