import json
import os
import sys
from datetime import datetime

import pytest

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


def test_run_job_persists_timestamps_in_utc(tmp_path):
    counter = {}
    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(_build_config(counter), state_file_path=str(state_path))

    assert manager.run_job("daily_pipeline", reason="scheduled") is True

    saved = json.loads(state_path.read_text(encoding="utf-8"))
    event = saved["events"][-1]
    assert event["started_at"].endswith("+00:00")
    assert event["finished_at"].endswith("+00:00")


def test_in_progress_job_is_skipped(tmp_path):
    """実行中のジョブは2回目の run_job でスキップされること"""
    state_path = tmp_path / "state.json"
    import threading

    barrier = threading.Barrier(2)
    released = threading.Event()

    def slow_func():
        barrier.wait()  # テストスレッドと同期
        released.wait()  # 解放まで待機

    config = {
        "slow_job": {
            "func": slow_func,
            "period": "daily",
            "day_of_week": "*",
            "hour": 8,
            "minute": 0,
            "max_executions_per_period": 5,
        }
    }
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))

    # 別スレッドでジョブを実行開始
    t = threading.Thread(target=lambda: manager.run_job("slow_job", "background"))
    t.start()
    barrier.wait()  # slow_func が実行開始するまで待つ

    # この時点でジョブは _in_progress_jobs にある → スキップされるはず
    result = manager.run_job("slow_job", reason="concurrent", force=True)
    # force=True でも in_progress は優先
    assert result is False

    released.set()
    t.join()


def test_weekly_period_key(tmp_path):
    """weekly ジョブの period_key が ISO 週番号形式になること"""
    from zoneinfo import ZoneInfo

    state_path = tmp_path / "state.json"
    config = {
        "weekly_job": {
            "func": lambda: None,
            "period": "weekly",
            "day_of_week": "sat",
            "hour": 10,
            "minute": 0,
        }
    }
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))
    tz = ZoneInfo("Asia/Tokyo")
    dt = datetime(2026, 1, 3, 10, 0, 0, tzinfo=tz)  # 2026-01-03 は土曜
    key = manager._build_period_key(config["weekly_job"], dt)
    assert key.startswith("2026-W")


def test_is_scheduled_day_wildcard(tmp_path):
    """day_of_week='*' のとき全日が対象になること"""
    from zoneinfo import ZoneInfo

    state_path = tmp_path / "state.json"
    config = {
        "any_day": {
            "func": lambda: None,
            "period": "daily",
            "day_of_week": "*",
            "hour": 8,
            "minute": 0,
        }
    }
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))
    tz = ZoneInfo("Asia/Tokyo")
    for day in range(7):
        dt = datetime(2026, 1, 5 + day, 10, 0, 0, tzinfo=tz)
        assert manager._is_scheduled_day(config["any_day"], dt) is True


def test_load_state_with_invalid_json(tmp_path):
    """不正な JSON の場合、空イベントリストで初期化されること"""
    state_path = tmp_path / "state.json"
    state_path.write_text("not valid json", encoding="utf-8")
    config = {"job": {"func": lambda: None, "period": "daily", "hour": 8, "minute": 0}}
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))
    assert manager._state == {"events": []}


def test_load_state_with_non_dict_state(tmp_path):
    """JSON がリスト形式の場合も空イベントリストで初期化されること"""
    state_path = tmp_path / "state.json"
    state_path.write_text('["not", "a", "dict"]', encoding="utf-8")
    config = {"job": {"func": lambda: None, "period": "daily", "hour": 8, "minute": 0}}
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))
    assert manager._state == {"events": []}


def test_run_job_error_propagates(tmp_path):
    """func が例外を出した場合、例外が再送出されること"""
    state_path = tmp_path / "state.json"

    def failing_func():
        raise ValueError("意図的なエラー")

    config = {
        "fail_job": {
            "func": failing_func,
            "period": "daily",
            "day_of_week": "*",
            "hour": 8,
            "minute": 0,
        }
    }
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))
    with pytest.raises(ValueError, match="意図的なエラー"):
        manager.run_job("fail_job", reason="test")
