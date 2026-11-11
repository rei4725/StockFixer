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
    assert gapped == {
        "monthly_multibagger",
        "weekly_rule_evaluation",
        "regime_leverage_weekly",
        "weekly_optimization",
        "daily_rule_signals",
        "weekly_model_training",
        "weekly_report",
    }


def test_monthly_multibagger_recovery_settings_are_unchanged():
    from run_scheduler import SCHEDULE_CONFIG

    config = SCHEDULE_CONFIG["monthly_multibagger"]
    assert config["max_executions_per_period"] == 3
    assert config["recovery_min_gap_minutes"] == 720


def test_regime_leverage_weekly_recovery_settings_are_unchanged():
    from run_scheduler import SCHEDULE_CONFIG

    config = SCHEDULE_CONFIG["regime_leverage_weekly"]
    assert config["max_executions_per_period"] == 12
    assert config["recovery_min_gap_minutes"] == 240


def test_weekly_optimization_recovery_settings():
    from run_scheduler import SCHEDULE_CONFIG

    config = SCHEDULE_CONFIG["weekly_optimization"]
    assert config["long_recovery"] is True
    assert config["max_executions_per_period"] == 2
    assert config["max_interrupted_reruns_per_period"] == 40
    assert config["recovery_min_gap_minutes"] == 720


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


OPTIMIZATION = "weekly_optimization"
# 2026-11-04 は水曜（ISO 週 2026-W45）。定時は 01:00
WEDNESDAY_RUN = datetime(2026, 11, 4, 1, 0, tzinfo=ZoneInfo("Asia/Tokyo"))


def _production_manager(tmp_path, job_id: str) -> SchedulerQueueManager:
    """本物の設定のまま、失敗で落ちるジョブのマネージャを返す。"""
    from run_scheduler import SCHEDULE_CONFIG

    config = dict(SCHEDULE_CONFIG[job_id])
    config["func"] = MagicMock(side_effect=NonRetryableJobError("失敗が線を超えた"))
    return SchedulerQueueManager({job_id: config}, state_file_path=str(tmp_path / "state.json"))


def _failed_optimization(manager: SchedulerQueueManager, now: datetime) -> None:
    with (
        patch("src.orchestration.scheduler_queue.ping_heartbeat"),
        patch.object(manager, "now", return_value=now),
    ):
        with pytest.raises(NonRetryableJobError):
            manager.run_job(OPTIMIZATION, reason="scheduled")


def _lost_run(
    manager: SchedulerQueueManager, job_id: str, period_key: str, started: datetime
) -> None:
    """前の起動のプロセスで消えた回（終わりの記録の無い started）を足す。"""
    manager._append_event(
        {
            "job_id": job_id,
            "reason": "recovery",
            "status": "started",
            "period_key": period_key,
            "run_id": f"lost-{started.isoformat()}",
            "boot_id": "previous-boot",
            "started_at": isoformat_utc(started),
        }
    )


@pytest.mark.parametrize(
    "minutes, expected",
    [
        pytest.param(5, False, id="next-poll"),
        pytest.param(719, False, id="just-before-the-gap-ends"),
        pytest.param(720, True, id="gap-has-passed"),
    ],
)
def test_weekly_optimization_waits_after_failed_run(tmp_path, minutes, expected):
    manager = _production_manager(tmp_path, OPTIMIZATION)
    # 定時の 30 分の遅れより後に終わった回で、間の判定だけを見る
    failed = WEDNESDAY_RUN + timedelta(hours=2)
    _failed_optimization(manager, failed)

    now = failed + timedelta(minutes=minutes)
    assert manager.should_recover(OPTIMIZATION, now=now) is expected


def test_weekly_optimization_recovers_right_after_a_lost_run(tmp_path):
    manager = _production_manager(tmp_path, OPTIMIZATION)
    started = WEDNESDAY_RUN + timedelta(hours=2)
    _lost_run(manager, OPTIMIZATION, "2026-W45", started)

    assert manager.should_recover(OPTIMIZATION, now=started + timedelta(minutes=5)) is True


def test_weekly_optimization_lost_run_does_not_restart_the_gap_of_a_failed_run(tmp_path):
    manager = _production_manager(tmp_path, OPTIMIZATION)
    _failed_optimization(manager, WEDNESDAY_RUN)
    _lost_run(manager, OPTIMIZATION, "2026-W45", WEDNESDAY_RUN + timedelta(minutes=720))

    now = WEDNESDAY_RUN + timedelta(minutes=725)
    assert manager.should_recover(OPTIMIZATION, now=now) is True


def test_weekly_optimization_lost_run_does_not_shorten_the_gap_of_a_failed_run(tmp_path):
    manager = _production_manager(tmp_path, OPTIMIZATION)
    _lost_run(manager, OPTIMIZATION, "2026-W45", WEDNESDAY_RUN)
    failed = WEDNESDAY_RUN + timedelta(hours=2)
    _failed_optimization(manager, failed)

    assert manager.should_recover(OPTIMIZATION, now=failed + timedelta(minutes=719)) is False
    assert manager.should_recover(OPTIMIZATION, now=failed + timedelta(minutes=720)) is True


def test_rule_evaluation_recovers_after_a_lost_run_without_the_gap(tmp_path):
    manager = _production_manager(tmp_path, RULE_EVALUATION)
    _lost_run(manager, RULE_EVALUATION, "2026-W44", SUNDAY_RUN)

    now = SUNDAY_RUN + timedelta(minutes=65)
    assert manager.should_recover(RULE_EVALUATION, now=now) is True


TRAINING = "weekly_model_training"
# 2026-11-03 は火曜（ISO 週 2026-W45）。定時は 01:00
TUESDAY_RUN = datetime(2026, 11, 3, 1, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
# 定時の 15 分の遅れより後に終わった回で、間の判定だけを見る（#1682）
TRAINING_FAILED = TUESDAY_RUN + timedelta(minutes=20)


def _failed_training(
    manager: SchedulerQueueManager, now: datetime, reason: str = "scheduled"
) -> None:
    with (
        patch("src.orchestration.scheduler_queue.ping_heartbeat"),
        patch.object(manager, "now", return_value=now),
    ):
        with pytest.raises(NonRetryableJobError):
            manager.run_job(TRAINING, reason=reason)


def test_weekly_training_recovery_settings():
    from run_scheduler import SCHEDULE_CONFIG

    config = SCHEDULE_CONFIG[TRAINING]
    assert config["max_executions_per_period"] == 2
    assert config["recovery_min_gap_minutes"] == 240
    assert "rerun_if_interrupted" not in config
    assert "recover_rest_of_week" not in config


@pytest.mark.parametrize(
    "minutes, expected",
    [
        pytest.param(5, False, id="next-poll"),
        pytest.param(239, False, id="just-before-the-gap-ends"),
        pytest.param(240, True, id="gap-has-passed"),
    ],
)
def test_weekly_training_waits_after_failed_run(tmp_path, minutes, expected):
    manager = _production_manager(tmp_path, TRAINING)
    _failed_training(manager, TRAINING_FAILED)

    now = TRAINING_FAILED + timedelta(minutes=minutes)
    assert manager.should_recover(TRAINING, now=now) is expected


def test_weekly_training_keeps_the_second_attempt_through_a_short_outage(tmp_path):
    manager = _production_manager(tmp_path, TRAINING)
    _failed_training(manager, TRAINING_FAILED)
    assert manager.should_recover(TRAINING, now=TRAINING_FAILED + timedelta(minutes=30)) is False

    recovery_at = TRAINING_FAILED + timedelta(minutes=240)
    assert manager.should_recover(TRAINING, now=recovery_at) is True
    _failed_training(manager, recovery_at, reason="recovery")

    assert manager.should_recover(TRAINING, now=recovery_at + timedelta(minutes=240)) is False
    assert manager.get_execution_count(TRAINING, "2026-W45") == 2


def test_weekly_training_does_not_recover_on_wednesday(tmp_path):
    manager = _production_manager(tmp_path, TRAINING)
    _failed_training(manager, TRAINING_FAILED)

    assert manager.should_recover(TRAINING, now=TRAINING_FAILED + timedelta(days=1)) is False


def test_weekly_training_waits_the_gap_after_a_lost_run(tmp_path):
    manager = _production_manager(tmp_path, TRAINING)
    _lost_run(manager, TRAINING, "2026-W45", TUESDAY_RUN)

    assert manager.should_recover(TRAINING, now=TUESDAY_RUN + timedelta(minutes=20)) is False
    assert manager.should_recover(TRAINING, now=TUESDAY_RUN + timedelta(minutes=239)) is False
    assert manager.should_recover(TRAINING, now=TUESDAY_RUN + timedelta(minutes=240)) is True


REPORT = "weekly_report"
# 2026-11-05 は木曜（ISO 週 2026-W45）。定時は 02:00
THURSDAY_RUN = datetime(2026, 11, 5, 2, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
# 定時の 15 分の遅れより後に終わった回で、間の判定だけを見る（#1689）
REPORT_FAILED = THURSDAY_RUN + timedelta(minutes=20)


def _failed_report(
    manager: SchedulerQueueManager, now: datetime, reason: str = "scheduled"
) -> None:
    with (
        patch("src.orchestration.scheduler_queue.ping_heartbeat"),
        patch.object(manager, "now", return_value=now),
    ):
        with pytest.raises(NonRetryableJobError):
            manager.run_job(REPORT, reason=reason)


def test_weekly_report_recovery_settings():
    from run_scheduler import SCHEDULE_CONFIG

    config = SCHEDULE_CONFIG[REPORT]
    assert config["max_executions_per_period"] == 2
    assert config["recovery_min_gap_minutes"] == 240
    assert config["rerun_if_interrupted"] is True
    assert "recover_rest_of_week" not in config


@pytest.mark.parametrize(
    "minutes, expected",
    [
        pytest.param(5, False, id="next-poll"),
        pytest.param(239, False, id="just-before-the-gap-ends"),
        pytest.param(240, True, id="gap-has-passed"),
    ],
)
def test_weekly_report_waits_after_failed_run(tmp_path, minutes, expected):
    manager = _production_manager(tmp_path, REPORT)
    _failed_report(manager, REPORT_FAILED)

    now = REPORT_FAILED + timedelta(minutes=minutes)
    assert manager.should_recover(REPORT, now=now) is expected


def test_weekly_report_keeps_the_second_attempt_through_a_short_outage(tmp_path):
    manager = _production_manager(tmp_path, REPORT)
    _failed_report(manager, REPORT_FAILED)
    assert manager.should_recover(REPORT, now=REPORT_FAILED + timedelta(minutes=5)) is False

    recovery_at = REPORT_FAILED + timedelta(minutes=240)
    assert manager.should_recover(REPORT, now=recovery_at) is True
    _failed_report(manager, recovery_at, reason="recovery")

    assert manager.should_recover(REPORT, now=recovery_at + timedelta(minutes=240)) is False
    assert manager.get_execution_count(REPORT, "2026-W45") == 2


def test_weekly_report_does_not_recover_on_friday(tmp_path):
    manager = _production_manager(tmp_path, REPORT)
    _failed_report(manager, REPORT_FAILED)

    assert manager.should_recover(REPORT, now=REPORT_FAILED + timedelta(days=1)) is False


def test_weekly_report_recovers_right_after_a_lost_run(tmp_path):
    manager = _production_manager(tmp_path, REPORT)
    _lost_run(manager, REPORT, "2026-W45", THURSDAY_RUN)

    assert manager.should_recover(REPORT, now=THURSDAY_RUN + timedelta(minutes=15)) is True


def _lock_expired_run(
    manager: SchedulerQueueManager, job_id: str, started: datetime, finished: datetime
) -> None:
    """ロックを待ち切って落ちた回を1回通す。"""
    from src.orchestration.job_errors import LockWaitExpiredJobError

    manager.schedule_config[job_id]["func"] = MagicMock(
        side_effect=LockWaitExpiredJobError("ロックが取れない")
    )
    with (
        patch("src.orchestration.scheduler_queue.ping_heartbeat"),
        patch.object(manager, "now", side_effect=[started, started, finished]),
    ):
        with pytest.raises(NonRetryableJobError):
            manager.run_job(job_id, reason="recovery")


def test_weekly_training_recovers_right_after_the_lock_wait_expired(tmp_path):
    manager = _production_manager(tmp_path, TRAINING)
    expired = TUESDAY_RUN + timedelta(hours=3)
    _lock_expired_run(manager, TRAINING, TUESDAY_RUN, expired)

    assert manager.should_recover(TRAINING, now=expired + timedelta(minutes=5)) is True


def test_weekly_training_stops_after_the_second_lock_wait_expired(tmp_path):
    manager = _production_manager(tmp_path, TRAINING)
    expired = TUESDAY_RUN + timedelta(hours=3)
    _lock_expired_run(manager, TRAINING, TUESDAY_RUN, expired)
    second = expired + timedelta(minutes=5)
    _lock_expired_run(manager, TRAINING, second, second + timedelta(hours=3))

    assert manager.should_recover(TRAINING, now=second + timedelta(hours=4)) is False
    assert manager.get_execution_count(TRAINING, "2026-W45") == 2


def test_failed_run_keeps_its_gap_after_a_lock_expired_run(tmp_path):
    manager = _manager(tmp_path, recovery_min_gap_minutes=120)
    failed = datetime(2026, 11, 1, 5, 0, tzinfo=manager.tz)
    _failed_run(manager, failed)
    expired = failed + timedelta(minutes=60)
    _lock_expired_run(manager, JOB, expired, expired)

    assert manager.should_recover(JOB, now=failed + timedelta(minutes=119)) is False
    assert manager.should_recover(JOB, now=failed + timedelta(minutes=120)) is True
