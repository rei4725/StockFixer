"""run_job の終わりの記録の書き込みが失敗したときの扱い（#1165）"""

import os
import sys
from unittest.mock import patch

import pytest

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.orchestration.scheduler_queue import SchedulerQueueManager  # noqa: E402


def _build_manager(tmp_path, func):
    config = {
        "daily_pipeline": {
            "func": func,
            "period": "daily",
            "day_of_week": "*",
            "hour": 8,
            "minute": 0,
            "max_retries": 0,
            "max_executions_per_period": 1,
        }
    }
    return SchedulerQueueManager(config, state_file_path=str(tmp_path / "state.json"))


def _fail_finish_save_once(manager):
    """終わりの記録（success・error）の _save_state だけを1回失敗させる。"""
    original_save = manager._save_state
    failed = {"n": 0}

    def _save_state():
        last = manager._state["events"][-1]
        if failed["n"] == 0 and last.get("status") in {"success", "error"}:
            failed["n"] += 1
            raise OSError("ディスクの空きがありません")
        original_save()

    manager._save_state = _save_state  # type: ignore[method-assign]
    return failed


@patch("src.orchestration.scheduler_queue.ping_heartbeat")
def test_finish_write_failure_releases_in_progress_mark(mock_ping, tmp_path):
    calls = {"n": 0}

    def _func():
        calls["n"] += 1

    manager = _build_manager(tmp_path, _func)
    failed = _fail_finish_save_once(manager)

    manager.run_job("daily_pipeline", reason="scheduled")

    assert failed["n"] == 1
    assert manager._in_progress_jobs == set()
    assert manager.run_job("daily_pipeline", reason="manual", force=True) is True
    assert calls["n"] == 2


@patch("src.orchestration.scheduler_queue.ping_heartbeat")
def test_finish_write_failure_keeps_original_job_exception(mock_ping, tmp_path, caplog):
    def _failing_func():
        raise ValueError("ジョブの元の例外")

    manager = _build_manager(tmp_path, _failing_func)
    _fail_finish_save_once(manager)

    with caplog.at_level("ERROR", logger="scheduler"):
        with pytest.raises(ValueError, match="ジョブの元の例外"):
            manager.run_job("daily_pipeline", reason="scheduled")

    write_errors = [
        r
        for r in caplog.records
        if r.levelname == "ERROR" and "終わりの記録" in r.getMessage() and r.exc_info
    ]
    assert len(write_errors) == 1
    assert isinstance(write_errors[0].exc_info[1], OSError)
    assert manager._in_progress_jobs == set()


@patch("src.orchestration.scheduler_queue.ping_heartbeat")
def test_finish_write_failure_still_pings_heartbeat(mock_ping, tmp_path):
    manager = _build_manager(tmp_path, lambda: None)
    _fail_finish_save_once(manager)

    manager.run_job("daily_pipeline", reason="scheduled")

    mock_ping.assert_called_once_with("daily_pipeline", success=True)
