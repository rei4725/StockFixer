"""前の起動のプロセスで消えた回（終わりの無い started）の補完（#946）"""

import json
import logging
import os
import sys
import threading
from datetime import datetime

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.orchestration.scheduler_queue import SchedulerQueueManager  # noqa: E402


def _write_started(state_path, job_id: str, period_key: str, run_id: str, boot_id: str) -> None:
    """前の起動のプロセスが始めの記録だけを残して消えた状態を作る"""
    events = []
    if state_path.exists():
        events = json.loads(state_path.read_text(encoding="utf-8"))["events"]
    events.append(
        {
            "job_id": job_id,
            "reason": "scheduled",
            "status": "started",
            "period_key": period_key,
            "run_id": run_id,
            "boot_id": boot_id,
            "started_at": "2020-01-01T00:00:00+00:00",
        }
    )
    state_path.write_text(json.dumps({"events": events}), encoding="utf-8")


def test_daily_backup_interrupted_by_previous_boot_is_recovered_today(tmp_path):
    """前の起動で started だけ残った daily_backup の当日分を補完してよいと返すこと"""
    from run_scheduler import SCHEDULE_CONFIG

    state_path = tmp_path / "state.json"
    _write_started(state_path, "daily_backup", "2026-10-20", "run-a", "previous-boot")
    manager = SchedulerQueueManager(SCHEDULE_CONFIG, state_file_path=str(state_path))

    # 03:30 + 60分を過ぎた時刻
    now = datetime(2026, 10, 20, 5, 0, tzinfo=manager.tz)
    assert manager.should_recover("daily_backup", now=now) is True
    assert manager.get_execution_count("daily_backup", "2026-10-20") == 0


def test_daily_backup_interrupted_by_previous_boot_is_backfilled(tmp_path):
    """前の起動で started だけ残った daily_backup の前日分を補完してよいと返すこと"""
    from run_scheduler import SCHEDULE_CONFIG

    state_path = tmp_path / "state.json"
    _write_started(state_path, "daily_backup", "2026-10-19", "run-a", "previous-boot")
    manager = SchedulerQueueManager(SCHEDULE_CONFIG, state_file_path=str(state_path))

    # 当日の 03:30 より前の起動
    now = datetime(2026, 10, 20, 2, 0, tzinfo=manager.tz)
    assert "2026-10-19" in manager.get_missed_past_periods("daily_backup", now=now)


def test_interrupted_run_logs_warning_once(tmp_path, caplog):
    """消えた回を見つけたら WARNING を出し、同じ回では出し直さないこと"""
    from run_scheduler import SCHEDULE_CONFIG

    state_path = tmp_path / "state.json"
    _write_started(state_path, "daily_backup", "2026-10-20", "run-a", "previous-boot")
    manager = SchedulerQueueManager(SCHEDULE_CONFIG, state_file_path=str(state_path))
    now = datetime(2026, 10, 20, 5, 0, tzinfo=manager.tz)

    with caplog.at_level(logging.WARNING, logger="scheduler"):
        manager.should_recover("daily_backup", now=now)
        manager.should_recover("daily_backup", now=now)
    warnings = [r for r in caplog.records if r.levelname == "WARNING" and "run-a" in r.message]
    assert len(warnings) == 1


@pytest.mark.parametrize("job_id", ["daily_auto_order", "daily_rule_signals"])
def test_order_jobs_interrupted_by_previous_boot_are_not_recovered(tmp_path, job_id):
    """発注のジョブは、前の起動で消えた回も今までどおり補完しないこと"""
    from run_scheduler import SCHEDULE_CONFIG

    assert not SCHEDULE_CONFIG[job_id].get("rerun_if_interrupted", False)
    state_path = tmp_path / "state.json"
    _write_started(state_path, job_id, "2026-10-20", "run-a", "previous-boot")
    manager = SchedulerQueueManager(SCHEDULE_CONFIG, state_file_path=str(state_path))

    # 2026-10-20 は火曜。daily_auto_order の締め 09:30 より前
    now = datetime(2026, 10, 20, 9, 10, tzinfo=manager.tz)
    assert manager.should_recover(job_id, now=now) is False
    assert manager.get_execution_count(job_id, "2026-10-20") == 1


def test_weekly_model_training_interrupted_by_previous_boot_is_counted(tmp_path):
    """週の学習（#807 の持ち場）は、前の起動で消えた回を今までどおり上限に数えること"""
    from run_scheduler import SCHEDULE_CONFIG

    assert not SCHEDULE_CONFIG["weekly_model_training"].get("rerun_if_interrupted", False)
    state_path = tmp_path / "state.json"
    _write_started(state_path, "weekly_model_training", "2026-W43", "run-a", "previous-boot")
    _write_started(state_path, "weekly_model_training", "2026-W43", "run-b", "previous-boot-2")
    manager = SchedulerQueueManager(SCHEDULE_CONFIG, state_file_path=str(state_path))

    # 2026-10-20 は火曜。01:00 + 15分を過ぎた時刻
    now = datetime(2026, 10, 20, 2, 0, tzinfo=manager.tz)
    assert manager.should_recover("weekly_model_training", now=now) is False
    assert manager.get_execution_count("weekly_model_training", "2026-W43") == 2


def test_interrupted_recovery_stops_after_two_consecutive_interruptions(tmp_path):
    """消えた回の補完は1期間に1回までで、2回続けて消えたら止めること（C-014）"""
    state_path = tmp_path / "state.json"
    calls = {"n": 0}
    snapshot: dict[str, bytes] = {}

    def _killed_func():
        calls["n"] += 1
        snapshot["state"] = state_path.read_bytes()

    config = {
        "daily_backup": {
            "func": _killed_func,
            "period": "daily",
            "day_of_week": "mon-sun",
            "hour": 3,
            "minute": 30,
            "recovery_delay_minutes": 60,
            "max_executions_per_period": 1,
            "max_retries": 0,
            "rerun_if_interrupted": True,
        }
    }
    tz = SchedulerQueueManager(config, state_file_path=str(state_path)).tz
    now = datetime(2026, 10, 20, 5, 0, tzinfo=tz)

    first = SchedulerQueueManager(config, state_file_path=str(state_path))
    first.now = lambda: now  # type: ignore[method-assign]
    first.run_job("daily_backup", reason="scheduled")
    state_path.write_bytes(snapshot["state"])

    for _ in range(10):
        # 起き直したプロセスとして状態ファイルを読み直す
        manager = SchedulerQueueManager(config, state_file_path=str(state_path))
        manager.now = lambda: now  # type: ignore[method-assign]
        if not manager.should_recover("daily_backup", now=now):
            break
        manager.run_job("daily_backup", reason="recovery")
        state_path.write_bytes(snapshot["state"])

    assert calls["n"] == 2
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))
    assert manager.get_execution_count("daily_backup", "2026-10-20") == 1


def test_run_in_progress_in_same_process_is_not_recovered(tmp_path):
    """同じプロセスで動いている回は、印のあるジョブでも補完しないこと"""
    started = threading.Event()
    released = threading.Event()

    def slow_func():
        started.set()
        released.wait()

    config = {
        "daily_backup": {
            "func": slow_func,
            "period": "daily",
            "day_of_week": "mon-sun",
            "hour": 3,
            "minute": 30,
            "recovery_delay_minutes": 60,
            "max_executions_per_period": 1,
            "rerun_if_interrupted": True,
        }
    }
    manager = SchedulerQueueManager(config, state_file_path=str(tmp_path / "state.json"))
    now = datetime(2026, 10, 20, 5, 0, tzinfo=manager.tz)
    manager.now = lambda: now  # type: ignore[method-assign]

    t = threading.Thread(target=lambda: manager.run_job("daily_backup", reason="scheduled"))
    t.start()
    try:
        assert started.wait(timeout=5)
        assert manager.should_recover("daily_backup", now=now) is False
        assert manager.get_execution_count("daily_backup", "2026-10-20") == 1
        assert manager.run_job("daily_backup", reason="recovery") is False
    finally:
        released.set()
        t.join()


def test_manual_run_in_another_process_is_not_recovered(tmp_path):
    """常駐の起動の後に別のプロセスが手で始めた途中の回は、補完しないこと"""
    started = threading.Event()
    released = threading.Event()

    def slow_func():
        started.set()
        released.wait()

    config = {
        "daily_backup": {
            "func": slow_func,
            "period": "daily",
            "day_of_week": "mon-sun",
            "hour": 3,
            "minute": 30,
            "recovery_delay_minutes": 60,
            "max_executions_per_period": 1,
            "rerun_if_interrupted": True,
        }
    }
    state_path = str(tmp_path / "state.json")
    resident = SchedulerQueueManager(config, state_file_path=state_path)
    manual = SchedulerQueueManager(config, state_file_path=state_path)
    now = datetime(2026, 10, 20, 5, 0, tzinfo=resident.tz)

    # 手での実行は常駐の起動より後に始まる（started_at は今の時刻で書かれる）
    t = threading.Thread(
        target=lambda: manual.run_job("daily_backup", reason="manual", period_key="2026-10-20")
    )
    t.start()
    try:
        assert started.wait(timeout=5)
        assert resident.should_recover("daily_backup", now=now) is False
        assert resident.get_execution_count("daily_backup", "2026-10-20") == 1
    finally:
        released.set()
        t.join()
