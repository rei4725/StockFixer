"""src/utils/health_check.py の単体テスト（#940）。"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from src.utils.health_check import is_degraded_for_daily, is_scheduler_stale


def _write_state(tmp_path, finished_at: str) -> None:
    state = {
        "events": [{"job_id": "daily_pipeline", "status": "success", "finished_at": finished_at}]
    }
    (tmp_path / "scheduler_queue_state.json").write_text(json.dumps(state), encoding="utf-8")


class TestIsSchedulerStale:
    def test_older_than_30_minutes_is_stale(self):
        now = datetime(2026, 10, 23, 0, 0, tzinfo=timezone.utc)
        runs = {"daily_pipeline": (now - timedelta(minutes=31)).isoformat()}
        assert is_scheduler_stale(runs, now=now) is True

    def test_within_30_minutes_is_not_stale(self):
        now = datetime(2026, 10, 23, 0, 0, tzinfo=timezone.utc)
        runs = {"daily_pipeline": (now - timedelta(minutes=10)).isoformat()}
        assert is_scheduler_stale(runs, now=now) is False

    def test_naive_timestamp_is_treated_as_utc(self):
        now = datetime(2026, 10, 23, 0, 0, tzinfo=timezone.utc)
        runs = {"daily_pipeline": "2026-10-22T23:00:00"}
        assert is_scheduler_stale(runs, now=now) is True

    def test_empty_or_unparsable_is_not_stale(self):
        assert is_scheduler_stale({}) is False
        assert is_scheduler_stale({"daily_pipeline": None}) is False
        assert is_scheduler_stale({"daily_pipeline": "not a time"}) is False


class TestIsDegradedForDaily:
    """日次の道の判定はスケジューラの鮮度を見ない（#940）。"""

    def test_old_scheduler_run_is_not_degraded_when_db_and_state_are_readable(self, tmp_path):
        old = (datetime.now(timezone.utc) - timedelta(minutes=48)).isoformat()
        _write_state(tmp_path, old)
        with patch("src.utils.health_check.check_db", return_value=("ok", None, None)):
            assert is_degraded_for_daily(str(tmp_path)) is False

    def test_db_busy_is_not_degraded(self, tmp_path):
        with patch("src.utils.health_check.check_db", return_value=("busy", None, None)):
            assert is_degraded_for_daily(str(tmp_path)) is False

    def test_db_error_is_degraded(self, tmp_path):
        with patch("src.utils.health_check.check_db", return_value=("error", "refused", None)):
            assert is_degraded_for_daily(str(tmp_path)) is True

    def test_unreadable_state_is_degraded(self, tmp_path):
        (tmp_path / "scheduler_queue_state.json").write_text("not json {{{", encoding="utf-8")
        with patch("src.utils.health_check.check_db", return_value=("ok", None, None)):
            assert is_degraded_for_daily(str(tmp_path)) is True

    def test_quarantined_state_without_body_is_degraded(self, tmp_path):
        (tmp_path / "scheduler_queue_state.json.corrupt-20261023T000000000000Z").write_text(
            "broken", encoding="utf-8"
        )
        with patch("src.utils.health_check.check_db", return_value=("ok", None, None)):
            assert is_degraded_for_daily(str(tmp_path)) is True

    def test_uses_results_dir_by_default(self, tmp_path):
        (tmp_path / "scheduler_queue_state.json").write_text("[1, 2]", encoding="utf-8")
        with (
            patch("src.utils.health_check.check_db", return_value=("ok", None, None)),
            patch("src.utils.health_check.get_results_dir", return_value=str(tmp_path)),
        ):
            assert is_degraded_for_daily() is True
