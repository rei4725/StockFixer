import os
import sys
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.services.scheduler_queue import SchedulerQueueManager  # noqa: E402


def _build_config(counter: dict[str, int]):
    def _daily_func():
        counter["daily"] = counter.get("daily", 0) + 1

    def _weekly_func():
        counter["weekly"] = counter.get("weekly", 0) + 1

    return {
        "daily_pipeline": {
            "func": _daily_func,
            "period": "daily",
            "day_of_week": "mon-fri",
            "hour": 19,
            "minute": 0,
            "recovery_delay_minutes": 10,
            "max_executions_per_period": 2,
        },
        "weekly_model_training": {
            "func": _weekly_func,
            "period": "weekly",
            "day_of_week": "sat",
            "hour": 3,
            "minute": 0,
            "recovery_delay_minutes": 15,
            "max_executions_per_period": 2,
        },
    }


def test_should_recover_true_when_schedule_passed_and_no_success(tmp_path):
    counter = {}
    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(_build_config(counter), state_file_path=str(state_path))

    # 2026-03-02 は月曜
    now = datetime(2026, 3, 2, 19, 20, tzinfo=manager.tz)

    assert manager.should_recover("daily_pipeline", now=now) is True


def test_should_recover_false_after_success_event(tmp_path):
    counter = {}
    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(_build_config(counter), state_file_path=str(state_path))

    # 2026-03-02 分として実行記録（period_key を明示して時刻のズレを回避）
    manager.run_job("daily_pipeline", reason="scheduled", period_key="2026-03-02")

    now = datetime(2026, 3, 2, 19, 20, tzinfo=manager.tz)
    assert manager.should_recover("daily_pipeline", now=now) is False


def test_run_job_respects_execution_limit(tmp_path):
    counter = {}
    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(_build_config(counter), state_file_path=str(state_path))

    assert manager.run_job("daily_pipeline", reason="scheduled") is True
    assert manager.run_job("daily_pipeline", reason="recovery") is True
    assert manager.run_job("daily_pipeline", reason="recovery") is False

    assert counter["daily"] == 2


def test_run_job_force_bypasses_execution_limit(tmp_path):
    counter = {}
    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(_build_config(counter), state_file_path=str(state_path))

    assert manager.run_job("daily_pipeline", reason="scheduled") is True
    assert manager.run_job("daily_pipeline", reason="recovery") is True
    assert manager.run_job("daily_pipeline", reason="manual", force=True) is True

    assert counter["daily"] == 3


def test_get_missed_past_periods_detects_missed_day(tmp_path):
    """前日分が未実行の場合に get_missed_past_periods が検出する"""
    counter = {}
    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(_build_config(counter), state_file_path=str(state_path))

    # 再起動後の早朝（当日 19:00 前）を模倣
    # 2026-03-03 (火) 08:00 → 前日 2026-03-02 (月) 分が未実行
    now = datetime(2026, 3, 3, 8, 0, tzinfo=manager.tz)
    missed = manager.get_missed_past_periods("daily_pipeline", lookback_days=2, now=now)
    assert "2026-03-02" in missed


def test_get_missed_past_periods_empty_when_already_run(tmp_path):
    """前日分が成功済みの場合は空リストを返す"""
    counter = {}
    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(_build_config(counter), state_file_path=str(state_path))

    # 前日 (2026-03-02) の実行を記録
    manager.run_job("daily_pipeline", reason="scheduled", period_key="2026-03-02")

    now = datetime(2026, 3, 3, 8, 0, tzinfo=manager.tz)
    missed = manager.get_missed_past_periods("daily_pipeline", lookback_days=2, now=now)
    assert "2026-03-02" not in missed


def test_get_missed_past_periods_skips_non_scheduled_days(tmp_path):
    """週末は対象外のためスキップされる"""
    counter = {}
    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(_build_config(counter), state_file_path=str(state_path))

    # 2026-03-09 (月) 08:00 → 前日 2026-03-08 (日) は mon-fri 対象外
    now = datetime(2026, 3, 9, 8, 0, tzinfo=manager.tz)
    missed = manager.get_missed_past_periods("daily_pipeline", lookback_days=1, now=now)
    assert "2026-03-08" not in missed
