"""失敗の回の後の補完に、前の回の終わりからの間を空ける設定のテスト（#1607）"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.orchestration.job_errors import NonRetryableJobError
from src.orchestration.scheduler_queue import SchedulerQueueManager
from src.utils.japan_time import isoformat_utc

JOB = "gapped_job"


def _config(**overrides: Any) -> dict[str, Any]:
    return {
        "func": MagicMock(side_effect=RuntimeError("boom")),
        "trigger": "cron",
        "period": "monthly",
        "day_of_week": "mon-sun",
        "day": 1,
        "hour": 4,
        "minute": 0,
        "recovery_delay_minutes": 60,
        "max_executions_per_period": 3,
        "max_retries": 0,
        **overrides,
    }


def _manager(tmp_path, **overrides: Any) -> SchedulerQueueManager:
    return SchedulerQueueManager(
        {JOB: _config(**overrides)}, state_file_path=str(tmp_path / "queue_state.json")
    )


def _failed_run(manager: SchedulerQueueManager, now: datetime) -> None:
    with (
        patch("src.orchestration.scheduler_queue.ping_heartbeat"),
        patch.object(manager, "now", return_value=now),
    ):
        with pytest.raises(RuntimeError):
            manager.run_job(JOB, reason="scheduled")


def test_gapped_job_waits_after_failed_run(tmp_path):
    manager = _manager(tmp_path, recovery_min_gap_minutes=120)
    finished = datetime(2026, 11, 1, 5, 0, tzinfo=manager.tz)
    _failed_run(manager, finished)

    just_after = finished + timedelta(minutes=5)
    assert manager.should_recover(JOB, now=just_after) is False
    assert manager.should_recover(JOB, now=finished + timedelta(minutes=119)) is False


def test_gapped_job_recovers_once_gap_has_passed(tmp_path):
    manager = _manager(tmp_path, recovery_min_gap_minutes=120)
    finished = datetime(2026, 11, 1, 5, 0, tzinfo=manager.tz)
    _failed_run(manager, finished)

    assert manager.should_recover(JOB, now=finished + timedelta(minutes=120)) is True


def test_gap_counts_from_the_latest_run(tmp_path):
    manager = _manager(tmp_path, recovery_min_gap_minutes=120)
    first = datetime(2026, 11, 1, 5, 0, tzinfo=manager.tz)
    second = first + timedelta(minutes=120)
    _failed_run(manager, first)
    _failed_run(manager, second)

    assert manager.should_recover(JOB, now=second + timedelta(minutes=5)) is False
    assert manager.should_recover(JOB, now=second + timedelta(minutes=120)) is True


def test_gap_counts_from_the_end_not_the_start_of_the_run(tmp_path):
    manager = _manager(tmp_path, recovery_min_gap_minutes=120)
    started = datetime(2026, 11, 1, 4, 0, tzinfo=manager.tz)
    finished = started + timedelta(minutes=90)
    with (
        patch("src.orchestration.scheduler_queue.ping_heartbeat"),
        patch.object(manager, "now", side_effect=[started, started, finished]),
    ):
        with pytest.raises(RuntimeError):
            manager.run_job(JOB, reason="scheduled")

    assert manager.should_recover(JOB, now=started + timedelta(minutes=125)) is False
    assert manager.should_recover(JOB, now=finished + timedelta(minutes=120)) is True


def test_gap_applies_after_a_run_lost_with_the_previous_process(tmp_path):
    manager = _manager(tmp_path, recovery_min_gap_minutes=120)
    started = datetime(2026, 11, 1, 5, 0, tzinfo=manager.tz)
    manager._append_event(
        {
            "job_id": JOB,
            "reason": "scheduled",
            "status": "started",
            "period_key": "2026-11",
            "run_id": "lost-run",
            "boot_id": "previous-boot",
            "started_at": isoformat_utc(started),
        }
    )

    assert manager.should_recover(JOB, now=started + timedelta(minutes=5)) is False
    assert manager.should_recover(JOB, now=started + timedelta(minutes=120)) is True


def test_gap_does_not_look_at_the_previous_period(tmp_path):
    manager = _manager(tmp_path, recovery_min_gap_minutes=120)
    _failed_run(manager, datetime(2026, 10, 31, 23, 0, tzinfo=manager.tz))

    assert manager.should_recover(JOB, now=datetime(2026, 11, 1, 5, 0, tzinfo=manager.tz)) is True


def test_gap_does_not_delay_a_missed_scheduled_run(tmp_path):
    manager = _manager(tmp_path, recovery_min_gap_minutes=120)

    assert manager.should_recover(JOB, now=datetime(2026, 11, 1, 5, 0, tzinfo=manager.tz)) is True


def test_unreadable_timestamp_does_not_block_recovery(tmp_path):
    manager = _manager(tmp_path, recovery_min_gap_minutes=120)
    manager._append_event(
        {
            "job_id": JOB,
            "status": "error",
            "period_key": "2026-11",
            "run_id": "broken",
            "finished_at": "not-a-time",
        }
    )

    assert manager.should_recover(JOB, now=datetime(2026, 11, 1, 5, 0, tzinfo=manager.tz)) is True


def test_job_without_gap_recovers_right_after_failed_run(tmp_path):
    manager = _manager(tmp_path)
    finished = datetime(2026, 11, 1, 5, 0, tzinfo=manager.tz)
    _failed_run(manager, finished)

    assert manager.should_recover(JOB, now=finished + timedelta(minutes=5)) is True


def test_only_the_named_jobs_have_the_gap():
    from run_scheduler import SCHEDULE_CONFIG

    gapped = {
        job_id for job_id, config in SCHEDULE_CONFIG.items() if "recovery_min_gap_minutes" in config
    }
    assert gapped == {"monthly_multibagger", "weekly_rule_evaluation"}


def test_monthly_multibagger_recovery_settings_are_unchanged():
    from run_scheduler import SCHEDULE_CONFIG

    config = SCHEDULE_CONFIG["monthly_multibagger"]
    assert config["max_executions_per_period"] == 3
    assert config["recovery_min_gap_minutes"] == 720


def test_weekly_optimization_recovery_settings_are_unchanged():
    from run_scheduler import SCHEDULE_CONFIG

    config = SCHEDULE_CONFIG["weekly_optimization"]
    assert config["long_recovery"] is True
    assert config["max_interrupted_reruns_per_period"] == 40
    assert "recovery_min_gap_minutes" not in config


def test_monthly_multibagger_keeps_attempts_through_a_short_outage(tmp_path):
    from run_scheduler import SCHEDULE_CONFIG

    config = dict(SCHEDULE_CONFIG["monthly_multibagger"])
    config["func"] = MagicMock(side_effect=RuntimeError("outage"))
    config["max_retries"] = 0
    manager = SchedulerQueueManager(
        {"monthly_multibagger": config}, state_file_path=str(tmp_path / "queue_state.json")
    )
    scheduled = datetime(2026, 11, 1, 4, 0, tzinfo=manager.tz)
    with (
        patch("src.orchestration.scheduler_queue.ping_heartbeat"),
        patch.object(manager, "now", return_value=scheduled),
    ):
        with pytest.raises(RuntimeError):
            manager.run_job("monthly_multibagger", reason="scheduled")

    assert (
        manager.should_recover("monthly_multibagger", now=scheduled + timedelta(hours=1)) is False
    )
    assert (
        manager.should_recover("monthly_multibagger", now=scheduled + timedelta(hours=12)) is True
    )


RULE_EVALUATION = "weekly_rule_evaluation"
# 2026-11-01 は日曜（ISO 週 2026-W44）
SUNDAY_RUN = datetime(2026, 11, 1, 2, 0, tzinfo=ZoneInfo("Asia/Tokyo"))


def _rule_evaluation_manager(tmp_path) -> SchedulerQueueManager:
    """本物の設定のまま、評価の失敗で落ちる週次ルール評価のマネージャを返す。"""
    from run_scheduler import SCHEDULE_CONFIG

    config = dict(SCHEDULE_CONFIG[RULE_EVALUATION])
    config["func"] = MagicMock(side_effect=NonRetryableJobError("評価の失敗が 3 件あった"))
    return SchedulerQueueManager(
        {RULE_EVALUATION: config}, state_file_path=str(tmp_path / "queue_state.json")
    )


def _failed_rule_evaluation(
    manager: SchedulerQueueManager, now: datetime, reason: str = "scheduled"
) -> None:
    with (
        patch("src.orchestration.scheduler_queue.ping_heartbeat"),
        patch.object(manager, "now", return_value=now),
    ):
        with pytest.raises(NonRetryableJobError):
            manager.run_job(RULE_EVALUATION, reason=reason)


def test_rule_evaluation_recovers_once_on_the_same_sunday(tmp_path):
    manager = _rule_evaluation_manager(tmp_path)
    _failed_rule_evaluation(manager, SUNDAY_RUN)

    assert manager.should_recover(RULE_EVALUATION, now=SUNDAY_RUN + timedelta(minutes=5)) is False
    assert manager.should_recover(RULE_EVALUATION, now=SUNDAY_RUN + timedelta(minutes=239)) is False
    recovery_at = SUNDAY_RUN + timedelta(minutes=240)
    assert manager.should_recover(RULE_EVALUATION, now=recovery_at) is True

    _failed_rule_evaluation(manager, recovery_at, reason="recovery")

    assert (
        manager.should_recover(RULE_EVALUATION, now=recovery_at + timedelta(minutes=240)) is False
    )
    assert manager.should_recover(RULE_EVALUATION, now=recovery_at + timedelta(hours=12)) is False


def test_rule_evaluation_failure_stays_a_single_call_within_the_run(tmp_path):
    manager = _rule_evaluation_manager(tmp_path)
    _failed_rule_evaluation(manager, SUNDAY_RUN)

    manager.schedule_config[RULE_EVALUATION]["func"].assert_called_once_with()


def test_rule_evaluation_does_not_recover_last_week_on_monday(tmp_path):
    manager = _rule_evaluation_manager(tmp_path)
    _failed_rule_evaluation(manager, SUNDAY_RUN)

    next_monday = datetime(2026, 11, 2, 7, 0, tzinfo=manager.tz)
    assert manager.should_recover(RULE_EVALUATION, now=next_monday) is False


def test_rule_evaluation_success_is_not_recovered(tmp_path):
    manager = _rule_evaluation_manager(tmp_path)
    manager.schedule_config[RULE_EVALUATION]["func"] = MagicMock()
    with (
        patch("src.orchestration.scheduler_queue.ping_heartbeat"),
        patch.object(manager, "now", return_value=SUNDAY_RUN),
    ):
        manager.run_job(RULE_EVALUATION, reason="scheduled")

    assert manager.should_recover(RULE_EVALUATION, now=SUNDAY_RUN + timedelta(hours=5)) is False
