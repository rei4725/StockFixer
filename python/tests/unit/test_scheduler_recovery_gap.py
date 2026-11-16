"""scheduler_recovery_gap の純な関数を events の並びだけで直に確かめる（#1854）。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from src.orchestration.scheduler_recovery_gap import _last_attempt_at, is_within_recovery_gap

JOB = "daily_job"
KEY = "2026-11-16"
BASE = datetime(2026, 11, 16, 0, 0, tzinfo=timezone.utc)
GAP = {"recovery_min_gap_minutes": 30}


def _error(
    finished_at: Any, *, run_id: str = "r1", job_id: str = JOB, key: str = KEY
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "period_key": key,
        "run_id": run_id,
        "status": "error",
        "finished_at": finished_at,
    }


def _started(started_at: Any, *, run_id: str = "r1") -> dict[str, Any]:
    return {
        "job_id": JOB,
        "period_key": KEY,
        "run_id": run_id,
        "status": "started",
        "started_at": started_at,
    }


class TestLastAttemptAt:
    def test_unreadable_stamp_is_skipped(self) -> None:
        events = [_error("not-a-time", run_id="r1"), _error(BASE.isoformat(), run_id="r2")]
        assert _last_attempt_at(events, JOB, KEY, include_started=True) == BASE

    def test_only_unreadable_stamp_gives_none(self) -> None:
        events = [_error("not-a-time")]
        assert _last_attempt_at(events, JOB, KEY, include_started=True) is None

    def test_non_string_stamp_is_skipped(self) -> None:
        events = [_error(BASE + timedelta(hours=1), run_id="r1"), _error(None, run_id="r2")]
        assert _last_attempt_at(events, JOB, KEY, include_started=True) is None

    def test_include_started_true_reads_started(self) -> None:
        later = BASE + timedelta(minutes=10)
        events = [_error(BASE.isoformat(), run_id="r1"), _started(later.isoformat(), run_id="r2")]
        assert _last_attempt_at(events, JOB, KEY, include_started=True) == later

    def test_include_started_false_ignores_started(self) -> None:
        later = BASE + timedelta(minutes=10)
        events = [_error(BASE.isoformat(), run_id="r1"), _started(later.isoformat(), run_id="r2")]
        assert _last_attempt_at(events, JOB, KEY, include_started=False) == BASE

    def test_include_started_false_with_only_started_gives_none(self) -> None:
        events = [_started(BASE.isoformat())]
        assert _last_attempt_at(events, JOB, KEY, include_started=False) is None

    def test_skip_recovery_gap_drops_both_start_and_end(self) -> None:
        start = _started(BASE.isoformat(), run_id="locked")
        end = _error((BASE + timedelta(minutes=5)).isoformat(), run_id="locked")
        end["skip_recovery_gap"] = True
        assert _last_attempt_at([start, end], JOB, KEY, include_started=True) is None

    def test_skip_recovery_gap_keeps_other_runs(self) -> None:
        start = _started((BASE + timedelta(minutes=20)).isoformat(), run_id="locked")
        end = _error((BASE + timedelta(minutes=25)).isoformat(), run_id="locked")
        end["skip_recovery_gap"] = True
        other = _error(BASE.isoformat(), run_id="r_other")
        assert _last_attempt_at([start, end, other], JOB, KEY, include_started=True) == BASE

    def test_other_job_and_other_period_are_ignored(self) -> None:
        events = [
            _error(BASE.isoformat(), job_id="other_job"),
            _error(BASE.isoformat(), key="2026-11-15"),
        ]
        assert _last_attempt_at(events, JOB, KEY, include_started=True) is None

    def test_success_is_not_counted(self) -> None:
        events = [
            {
                "job_id": JOB,
                "period_key": KEY,
                "run_id": "r1",
                "status": "success",
                "finished_at": BASE.isoformat(),
            }
        ]
        assert _last_attempt_at(events, JOB, KEY, include_started=True) is None

    def test_returns_latest_of_several(self) -> None:
        later = BASE + timedelta(minutes=40)
        events = [_error(later.isoformat(), run_id="r2"), _error(BASE.isoformat(), run_id="r1")]
        assert _last_attempt_at(events, JOB, KEY, include_started=True) == later

    def test_naive_and_z_stamps_are_read_as_utc(self) -> None:
        events = [_error("2026-11-16T00:00:00Z", run_id="r1"), _error("2026-11-16T00:00:00")]
        assert _last_attempt_at(events, JOB, KEY, include_started=True) == BASE


class TestIsWithinRecoveryGap:
    def test_job_without_gap_field_is_false(self) -> None:
        events = [_error(BASE.isoformat())]
        assert is_within_recovery_gap({}, events, JOB, KEY, BASE) is False

    def test_no_attempt_is_false(self) -> None:
        assert is_within_recovery_gap(GAP, [], JOB, KEY, BASE) is False

    def test_inside_gap_is_true(self) -> None:
        events = [_error(BASE.isoformat())]
        now = BASE + timedelta(minutes=29, seconds=59)
        assert is_within_recovery_gap(GAP, events, JOB, KEY, now) is True

    def test_boundary_now_equal_last_plus_gap_is_false(self) -> None:
        events = [_error(BASE.isoformat())]
        now = BASE + timedelta(minutes=30)
        assert is_within_recovery_gap(GAP, events, JOB, KEY, now) is False

    def test_after_gap_is_false(self) -> None:
        events = [_error(BASE.isoformat())]
        now = BASE + timedelta(minutes=31)
        assert is_within_recovery_gap(GAP, events, JOB, KEY, now) is False

    def test_without_rerun_if_interrupted_counts_started(self) -> None:
        events = [_started(BASE.isoformat())]
        now = BASE + timedelta(minutes=10)
        assert is_within_recovery_gap(GAP, events, JOB, KEY, now) is True

    def test_with_rerun_if_interrupted_ignores_started(self) -> None:
        config = {**GAP, "rerun_if_interrupted": True}
        events = [_started(BASE.isoformat())]
        now = BASE + timedelta(minutes=10)
        assert is_within_recovery_gap(config, events, JOB, KEY, now) is False

    def test_with_rerun_if_interrupted_still_counts_error(self) -> None:
        config = {**GAP, "rerun_if_interrupted": True}
        events = [_error(BASE.isoformat())]
        now = BASE + timedelta(minutes=10)
        assert is_within_recovery_gap(config, events, JOB, KEY, now) is True

    def test_skip_recovery_gap_run_does_not_hold_back(self) -> None:
        start = _started(BASE.isoformat(), run_id="locked")
        end = _error(BASE.isoformat(), run_id="locked")
        end["skip_recovery_gap"] = True
        now = BASE + timedelta(minutes=1)
        assert is_within_recovery_gap(GAP, [start, end], JOB, KEY, now) is False

    def test_unreadable_stamp_does_not_hold_back(self) -> None:
        events = [_error("garbage")]
        now = BASE + timedelta(minutes=1)
        assert is_within_recovery_gap(GAP, events, JOB, KEY, now) is False
