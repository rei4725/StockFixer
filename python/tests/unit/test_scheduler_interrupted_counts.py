"""scheduler_interrupted の純な関数を events の並びだけで直に確かめる（#1900）。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from src.orchestration.scheduler_interrupted import (
    PeriodCalendar,
    count_executions,
    is_backfill_daily,
    latest_missed_past_period,
)

JOB = "daily_job"
KEY = "2026-11-16"
BOOT_ID = "boot-now"
OLD_BOOT = "boot-old"
BOOTED_AT = datetime(2026, 11, 16, 9, 0, tzinfo=timezone.utc)
BEFORE = (BOOTED_AT - timedelta(hours=1)).isoformat()
AFTER = (BOOTED_AT + timedelta(hours=1)).isoformat()


def _event(
    status: str,
    run_id: str | None,
    *,
    job_id: str = JOB,
    key: str = KEY,
    boot_id: str | None = None,
    started_at: Any = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {"job_id": job_id, "period_key": key, "status": status}
    if run_id is not None:
        event["run_id"] = run_id
    if boot_id is not None:
        event["boot_id"] = boot_id
    if started_at is not None:
        event["started_at"] = started_at
    return event


def _count(events: list[dict[str, Any]], *, rerun: bool, max_forgiven: int = 1) -> Any:
    return count_executions(
        events,
        JOB,
        KEY,
        boot_id=BOOT_ID,
        booted_at=BOOTED_AT,
        rerun_if_interrupted=rerun,
        max_forgiven=max_forgiven,
    )


class TestCountExecutions:
    def test_success_and_error_are_counted(self) -> None:
        events = [_event("success", "r1"), _event("error", "r2")]
        result = _count(events, rerun=True)
        assert result.count == 2
        assert result.interrupted_run_ids == []

    def test_other_job_and_period_are_ignored(self) -> None:
        events = [
            _event("success", "r1", job_id="other"),
            _event("success", "r2", key="2026-11-15"),
        ]
        assert _count(events, rerun=False).count == 0

    def test_finished_started_is_not_counted_twice(self) -> None:
        events = [
            _event("started", "r1", boot_id=OLD_BOOT, started_at=BEFORE),
            _event("success", "r1"),
        ]
        result = _count(events, rerun=True)
        assert result.count == 1
        assert result.interrupted_run_ids == []

    def test_started_without_run_id_is_ignored(self) -> None:
        events = [_event("started", None, boot_id=OLD_BOOT, started_at=BEFORE)]
        assert _count(events, rerun=False).count == 0

    def test_interrupted_run_is_forgiven_with_rerun(self) -> None:
        events = [_event("started", "r1", boot_id=OLD_BOOT, started_at=BEFORE)]
        result = _count(events, rerun=True)
        assert result.count == 0
        assert result.interrupted_run_ids == ["r1"]

    def test_interrupted_run_is_counted_without_rerun(self) -> None:
        events = [_event("started", "r1", boot_id=OLD_BOOT, started_at=BEFORE)]
        result = _count(events, rerun=False)
        assert result.count == 1
        assert result.interrupted_run_ids == ["r1"]

    def test_forgiven_is_capped_by_max_forgiven(self) -> None:
        events = [
            _event("started", "r1", boot_id=OLD_BOOT, started_at=BEFORE),
            _event("started", "r2", boot_id=OLD_BOOT, started_at=BEFORE),
            _event("started", "r3", boot_id=OLD_BOOT, started_at=BEFORE),
        ]
        one = _count(events, rerun=True, max_forgiven=1)
        assert one.count == 2
        assert one.interrupted_run_ids == ["r1", "r2", "r3"]
        two = _count(events, rerun=True, max_forgiven=2)
        assert two.count == 1

    def test_started_without_boot_id_is_counted_but_not_interrupted(self) -> None:
        events = [_event("started", "r1", started_at=BEFORE)]
        result = _count(events, rerun=True)
        assert result.count == 1
        assert result.interrupted_run_ids == []

    def test_same_boot_is_not_interrupted(self) -> None:
        events = [_event("started", "r1", boot_id=BOOT_ID, started_at=BEFORE)]
        result = _count(events, rerun=True)
        assert result.count == 1
        assert result.interrupted_run_ids == []

    def test_started_after_boot_is_not_interrupted(self) -> None:
        events = [_event("started", "r1", boot_id=OLD_BOOT, started_at=AFTER)]
        result = _count(events, rerun=True)
        assert result.count == 1
        assert result.interrupted_run_ids == []

    def test_unreadable_started_at_is_not_interrupted(self) -> None:
        events = [
            _event("started", "r1", boot_id=OLD_BOOT, started_at="not-a-time"),
            _event("started", "r2", boot_id=OLD_BOOT, started_at=12345),
            _event("started", "r3", boot_id=OLD_BOOT),
        ]
        result = _count(events, rerun=True)
        assert result.count == 3
        assert result.interrupted_run_ids == []


class TestIsBackfillDaily:
    def test_daily_with_backfill(self) -> None:
        assert is_backfill_daily({"period": "daily", "backfill": True}) is True

    def test_period_defaults_to_daily(self) -> None:
        assert is_backfill_daily({"backfill": True}) is True

    def test_without_backfill(self) -> None:
        assert is_backfill_daily({"period": "daily"}) is False
        assert is_backfill_daily({"period": "daily", "backfill": False}) is False

    def test_weekly_with_backfill(self) -> None:
        assert is_backfill_daily({"period": "weekly", "backfill": True}) is False


# 2026-11-16 は月曜日。平日だけ動く日次のジョブとする
NOW = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
WEEKDAYS = {0, 1, 2, 3, 4}


def _calendar(counts: dict[str, int] | None = None) -> PeriodCalendar:
    table = counts or {}
    return PeriodCalendar(
        weekdays=WEEKDAYS,
        is_scheduled_day=lambda dt: dt.weekday() in WEEKDAYS,
        period_key_at=lambda dt: dt.strftime("%Y-%m-%d"),
        count_at=lambda key: table.get(key, 0),
    )


def _success(key: str) -> dict[str, Any]:
    return _event("success", "s-" + key, key=key)


class TestLatestMissedPastPeriod:
    CONFIG: dict[str, Any] = {"period": "daily", "backfill": True}

    def test_returns_latest_missed_scheduled_day(self) -> None:
        now = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)
        result = latest_missed_past_period(self.CONFIG, [], JOB, now, 2, _calendar())
        assert result == "2026-11-17"

    def test_skips_non_scheduled_days(self) -> None:
        # 月曜から2日遡ると日曜・土曜で、どちらも休みの曜日
        assert latest_missed_past_period(self.CONFIG, [], JOB, NOW, 2, _calendar()) is None
        # 3日遡ると金曜に届く
        assert latest_missed_past_period(self.CONFIG, [], JOB, NOW, 3, _calendar()) == "2026-11-13"

    def test_skips_day_covered_by_later_success(self) -> None:
        now = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)
        events = [_success("2026-11-18")]
        result = latest_missed_past_period(self.CONFIG, events, JOB, now, 2, _calendar())
        assert result is None

    def test_success_of_earlier_day_does_not_cover(self) -> None:
        now = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)
        events = [_success("2026-11-17")]
        result = latest_missed_past_period(self.CONFIG, events, JOB, now, 2, _calendar())
        assert result == "2026-11-18"

    def test_skips_day_at_max_executions(self) -> None:
        now = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)
        calendar = _calendar({"2026-11-18": 1})
        result = latest_missed_past_period(self.CONFIG, [], JOB, now, 2, calendar)
        assert result == "2026-11-17"

    def test_max_executions_from_config(self) -> None:
        now = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)
        config = {**self.CONFIG, "max_executions_per_period": 2}
        calendar = _calendar({"2026-11-18": 1})
        result = latest_missed_past_period(config, [], JOB, now, 2, calendar)
        assert result == "2026-11-18"

    def test_count_from_events_feeds_max_executions(self) -> None:
        # 前の起動で消えた回は rerun_if_interrupted で数えず、補完の対象に残る
        now = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)
        key = "2026-11-18"
        events = [_event("started", "r1", key=key, boot_id=OLD_BOOT, started_at=BEFORE)]

        def count_at(period_key: str) -> int:
            return count_executions(
                events,
                JOB,
                period_key,
                boot_id=BOOT_ID,
                booted_at=BOOTED_AT,
                rerun_if_interrupted=True,
            ).count

        calendar = PeriodCalendar(
            weekdays=WEEKDAYS,
            is_scheduled_day=lambda dt: dt.weekday() in WEEKDAYS,
            period_key_at=lambda dt: dt.strftime("%Y-%m-%d"),
            count_at=count_at,
        )
        result = latest_missed_past_period(self.CONFIG, events, JOB, now, 2, calendar)
        assert result == key
