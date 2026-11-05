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


def test_auto_order_interrupted_by_previous_boot_is_not_recovered(tmp_path):
    """8:50 の自動発注は、前の起動で消えた回も今までどおり補完しないこと

    16:30 のルールの発注は補完する（#1652。test_rule_signals_interrupted_rerun.py）。
    """
    from run_scheduler import SCHEDULE_CONFIG

    job_id = "daily_auto_order"
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


@pytest.mark.parametrize(
    "interrupted, expected_recover, expected_count",
    [(2, True, 0), (40, True, 0), (41, True, 1), (42, False, 2)],
)
def test_weekly_optimization_recovers_up_to_configured_limit(
    tmp_path, interrupted, expected_recover, expected_count
):
    """週の最適化の消えた回は設定の上限まで数えず、超えた分は数えて補完を止めること（#1522）

    回の上限は失敗の回の補完のため 2 回である（#1581）。
    """
    from run_scheduler import SCHEDULE_CONFIG

    assert SCHEDULE_CONFIG["weekly_optimization"]["max_executions_per_period"] == 2
    assert SCHEDULE_CONFIG["weekly_optimization"]["max_interrupted_reruns_per_period"] == 40
    state_path = tmp_path / "state.json"
    for i in range(interrupted):
        _write_started(state_path, "weekly_optimization", "2026-W45", f"run-{i}", f"boot-{i}")
    manager = SchedulerQueueManager(SCHEDULE_CONFIG, state_file_path=str(state_path))

    # 2026-11-04 は水曜。01:00 + 30分を過ぎた時刻
    now = datetime(2026, 11, 4, 2, 0, tzinfo=manager.tz)
    assert manager.should_recover("weekly_optimization", now=now) is expected_recover
    assert manager.get_execution_count("weekly_optimization", "2026-W45") == expected_count


def test_weekly_optimization_finished_run_still_counts_against_limit(tmp_path):
    """終わりまで走って落ちた回は、消えた回の上限の外で今までどおり数えること（#1522）"""
    from run_scheduler import SCHEDULE_CONFIG

    state_path = tmp_path / "state.json"
    for i in range(3):
        _write_started(state_path, "weekly_optimization", "2026-W45", f"run-{i}", f"boot-{i}")
    events = json.loads(state_path.read_text(encoding="utf-8"))["events"]
    for run_id in ("run-failed-1", "run-failed-2"):
        events.append(
            {
                "job_id": "weekly_optimization",
                "status": "error",
                "period_key": "2026-W45",
                "run_id": run_id,
            }
        )
    state_path.write_text(json.dumps({"events": events}), encoding="utf-8")
    manager = SchedulerQueueManager(SCHEDULE_CONFIG, state_file_path=str(state_path))

    now = datetime(2026, 11, 4, 2, 0, tzinfo=manager.tz)
    assert manager.get_execution_count("weekly_optimization", "2026-W45") == 2
    assert manager.should_recover("weekly_optimization", now=now) is False


def test_weekly_optimization_stops_after_limit_of_interruptions(tmp_path):
    """消えた回の許しの40回と回の上限の2回で計42回動き、それ以上は補完しないこと（#1534・#1581）"""
    from run_scheduler import SCHEDULE_CONFIG

    state_path = tmp_path / "state.json"
    calls = {"n": 0}
    snapshot: dict[str, bytes] = {}

    def _killed_func():
        calls["n"] += 1
        snapshot["state"] = state_path.read_bytes()

    config = {
        "weekly_optimization": {
            **SCHEDULE_CONFIG["weekly_optimization"],
            "func": _killed_func,
            "max_retries": 0,
        }
    }
    tz = SchedulerQueueManager(config, state_file_path=str(state_path)).tz
    now = datetime(2026, 11, 4, 2, 0, tzinfo=tz)

    first = SchedulerQueueManager(config, state_file_path=str(state_path))
    first.now = lambda: now  # type: ignore[method-assign]
    first.run_job("weekly_optimization", reason="scheduled")
    state_path.write_bytes(snapshot["state"])

    for _ in range(60):
        manager = SchedulerQueueManager(config, state_file_path=str(state_path))
        manager.now = lambda: now  # type: ignore[method-assign]
        if not manager.should_recover("weekly_optimization", now=now):
            break
        manager.run_job("weekly_optimization", reason="recovery")
        state_path.write_bytes(snapshot["state"])

    assert calls["n"] == 42


def test_only_weekly_optimization_has_more_interrupted_reruns(tmp_path):
    """ほかのジョブは消えた回の補完が1回までのままであること（#1522）"""
    from run_scheduler import SCHEDULE_CONFIG

    raised = [
        job_id
        for job_id, config in SCHEDULE_CONFIG.items()
        if "max_interrupted_reruns_per_period" in config
    ]
    assert raised == ["weekly_optimization"]

    state_path = tmp_path / "state.json"
    _write_started(state_path, "weekly_walk_forward_report", "2026-W45", "run-a", "boot-a")
    _write_started(state_path, "weekly_walk_forward_report", "2026-W45", "run-b", "boot-b")
    manager = SchedulerQueueManager(SCHEDULE_CONFIG, state_file_path=str(state_path))

    # 2026-11-05 は木曜。01:00 + 30分を過ぎた時刻
    now = datetime(2026, 11, 5, 2, 0, tzinfo=manager.tz)
    assert manager.should_recover("weekly_walk_forward_report", now=now) is False
    assert manager.get_execution_count("weekly_walk_forward_report", "2026-W45") == 1


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
