"""運用アラートの連続回数（NF-303-2・3・4）が同じ JST の日に二重に数えられないことのテスト（#1242）。"""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

from src.utils.alert_service import (
    update_drift_warn_streak,
    update_health_degraded_streak,
    update_loss_limit_streak,
)

_KEY_LOSS = "alert.loss_limit_streak"
_KEY_DRIFT = "alert.drift_warn_streak"
_KEY_HEALTH = "alert.health_degraded_streak"

_DAY1 = "20261026"
_DAY2 = "20261027"
_DAY3 = "20261028"


class _ConfigStoreTestCase(unittest.TestCase):
    """system_config を辞書に置き換える。"""

    def setUp(self):
        self.store: dict[str, str] = {}
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            patch(
                "src.utils.db.system_config.get_config_value",
                side_effect=lambda key, default=None: self.store.get(key, default),
            )
        )
        stack.enter_context(
            patch(
                "src.utils.db.system_config.set_config_value",
                side_effect=lambda key, value: self.store.__setitem__(key, value),
            )
        )


class _SameDayStreakCases:
    """3 つのストリークに共通の場合分け。サブクラスが update と key を決める。"""

    key: str

    def update(self, hit: bool, day_key: str) -> int:
        raise NotImplementedError

    def test_same_day_second_call_does_not_add(self):
        self.update(True, _DAY1)
        self.update(True, _DAY2)
        self.assertEqual(self.update(True, _DAY2), 2)
        self.assertEqual(self.store[self.key], "2")

    def test_same_day_rerun_reapplies_judgement_on_previous_value(self):
        self.update(True, _DAY1)
        self.update(True, _DAY2)
        self.assertEqual(self.update(False, _DAY2), 0)
        self.assertEqual(self.update(True, _DAY2), 2)

    def test_next_day_counts_up(self):
        self.update(True, _DAY1)
        self.update(True, _DAY1)
        self.assertEqual(self.update(True, _DAY2), 2)
        self.assertEqual(self.update(True, _DAY3), 3)

    def test_next_day_resets(self):
        self.update(True, _DAY1)
        self.update(True, _DAY2)
        self.assertEqual(self.update(False, _DAY3), 0)
        self.assertEqual(self.store[self.key], "0")

    def test_existing_streak_without_day_counts_up(self):
        # 日付の鍵が無い（直す前に保存された）値は、前の日までの値として数える
        self.store[self.key] = "1"
        self.assertEqual(self.update(True, _DAY2), 2)
        self.assertEqual(self.update(True, _DAY2), 2)


class TestDriftWarnStreakSameDay(_SameDayStreakCases, _ConfigStoreTestCase):
    key = _KEY_DRIFT

    def update(self, hit: bool, day_key: str) -> int:
        return update_drift_warn_streak(hit, day_key)


class TestLossLimitStreakSameDay(_SameDayStreakCases, _ConfigStoreTestCase):
    key = _KEY_LOSS

    def update(self, hit: bool, day_key: str) -> int:
        return update_loss_limit_streak(hit, day_key)


class TestHealthDegradedStreakSameDay(_SameDayStreakCases, _ConfigStoreTestCase):
    key = _KEY_HEALTH

    def update(self, hit: bool, day_key: str) -> int:
        return update_health_degraded_streak(hit, day_key)


class TestRecordStreakUsesJstDay(_ConfigStoreTestCase):
    """日次の道の記録が JST の今日を鍵に使い、やり直しの回で二重に数えない。"""

    def test_drift_rerun_same_day_counts_once(self):
        from src.orchestration.jobs import alert_streaks

        with patch.object(alert_streaks, "today_jst_key", return_value=_DAY1, create=True):
            alert_streaks.record_drift_warn_streak(True)
        with patch.object(alert_streaks, "today_jst_key", return_value=_DAY2, create=True):
            alert_streaks.record_drift_warn_streak(True)
            alert_streaks.record_drift_warn_streak(True)
        self.assertEqual(self.store[_KEY_DRIFT], "2")

    def test_loss_limit_rerun_same_day_counts_once(self):
        from src.orchestration.jobs import alert_streaks

        stats = {"reason_code": "daily_loss_limit"}
        with patch.object(alert_streaks, "today_jst_key", return_value=_DAY2, create=True):
            alert_streaks.record_loss_limit_streak(stats)
            alert_streaks.record_loss_limit_streak(stats)
        self.assertEqual(self.store[_KEY_LOSS], "1")

    def test_health_rerun_same_day_counts_once(self):
        from src.orchestration.jobs import alert_streaks

        with patch.object(alert_streaks, "today_jst_key", return_value=_DAY2), patch.object(
            alert_streaks, "is_degraded_for_daily", return_value=True
        ):
            alert_streaks.record_health_degraded_streak()
            alert_streaks.record_health_degraded_streak()
        self.assertEqual(self.store[_KEY_HEALTH], "1")


if __name__ == "__main__":
    unittest.main()
