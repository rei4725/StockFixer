"""定時の起動も締め（recovery_until）を過ぎて走り出したら動かさないことのテスト（#1298）"""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.orchestration.scheduler_queue import SchedulerQueueManager  # noqa: E402

_NOTIFY = "src.reporting.discord.discord_utils.send_webhook_text"


def _manager_with_counting_funcs(tmp_path, counter: dict[str, int]) -> SchedulerQueueManager:
    """本番の SCHEDULE_CONFIG の時刻と締めのまま、ジョブの関数だけを数える関数に替える"""
    from run_scheduler import SCHEDULE_CONFIG

    config = {}
    for job_id, job_config in SCHEDULE_CONFIG.items():

        def _func(_job_id=job_id):
            counter[_job_id] = counter.get(_job_id, 0) + 1

        config[job_id] = {**job_config, "func": _func}
    return SchedulerQueueManager(config, state_file_path=str(tmp_path / "state.json"))


def test_scheduled_auto_order_started_after_cutoff_does_not_order(tmp_path, caplog):
    """8:50 の定時の回が 9:31 に走り出したら、発注せずに WARNING と知らせを出して飛ばす"""
    counter: dict[str, int] = {}
    manager = _manager_with_counting_funcs(tmp_path, counter)
    # 2026-03-02 は月曜
    started = datetime(2026, 3, 2, 9, 31, tzinfo=manager.tz)
    manager.now = lambda: started  # type: ignore[method-assign]

    with patch(_NOTIFY) as mock_notify, caplog.at_level("WARNING", logger="scheduler"):
        assert manager.run_job("daily_auto_order", reason="scheduled") is False

    assert counter.get("daily_auto_order", 0) == 0
    assert any(
        r.levelname == "WARNING" and "daily_auto_order" in r.getMessage() for r in caplog.records
    )
    mock_notify.assert_called_once()
    assert "daily_auto_order" in mock_notify.call_args[0][0]


def test_scheduled_auto_order_started_after_cutoff_in_utc_still_skips(tmp_path):
    """時刻を UTC で測っても、JST に直して締めと比べる（コンテナの時刻は UTC）"""
    counter: dict[str, int] = {}
    manager = _manager_with_counting_funcs(tmp_path, counter)
    # 2026-03-02 00:31 UTC = 09:31 JST
    started = datetime(2026, 3, 2, 0, 31, tzinfo=timezone.utc)
    manager.now = lambda: started  # type: ignore[method-assign]

    with patch(_NOTIFY):
        assert manager.run_job("daily_auto_order", reason="scheduled") is False
    assert counter.get("daily_auto_order", 0) == 0


@pytest.mark.parametrize("hour, minute", [(8, 50), (9, 29)])
def test_scheduled_auto_order_before_cutoff_still_orders(tmp_path, hour, minute):
    """締めの前に走り出した定時の回は、今までどおり発注の処理に進む"""
    counter: dict[str, int] = {}
    manager = _manager_with_counting_funcs(tmp_path, counter)
    started = datetime(2026, 3, 2, hour, minute, tzinfo=manager.tz)
    manager.now = lambda: started  # type: ignore[method-assign]

    with patch(_NOTIFY) as mock_notify:
        assert manager.run_job("daily_auto_order", reason="scheduled") is True

    assert counter["daily_auto_order"] == 1
    mock_notify.assert_not_called()


@pytest.mark.parametrize("job_id", ["daily_pipeline", "daily_settle_orders", "daily_rule_signals"])
def test_scheduled_jobs_without_cutoff_run_late(tmp_path, job_id):
    """recovery_until の無いジョブの定時の起動は、遅れても今までどおり動く"""
    from run_scheduler import SCHEDULE_CONFIG

    assert "recovery_until" not in SCHEDULE_CONFIG[job_id]
    counter: dict[str, int] = {}
    manager = _manager_with_counting_funcs(tmp_path, counter)
    started = datetime(2026, 3, 2, 23, 0, tzinfo=manager.tz)
    manager.now = lambda: started  # type: ignore[method-assign]

    with patch(_NOTIFY) as mock_notify:
        assert manager.run_job(job_id, reason="scheduled") is True

    assert counter[job_id] == 1
    mock_notify.assert_not_called()


@pytest.mark.parametrize("reason", ["manual", "recovery"])
def test_cutoff_gate_is_only_for_scheduled_start(tmp_path, reason):
    """手での即時実行（force=True）は対象の外。補完の道の締めは should_recover が見る"""
    counter: dict[str, int] = {}
    manager = _manager_with_counting_funcs(tmp_path, counter)
    started = datetime(2026, 3, 2, 9, 31, tzinfo=manager.tz)
    manager.now = lambda: started  # type: ignore[method-assign]

    with patch(_NOTIFY) as mock_notify:
        assert manager.run_job("daily_auto_order", reason=reason, force=True) is True

    assert counter["daily_auto_order"] == 1
    mock_notify.assert_not_called()


def test_cutoff_skip_survives_notify_failure(tmp_path):
    """知らせが落ちても、発注せずに飛ばす"""
    counter: dict[str, int] = {}
    manager = _manager_with_counting_funcs(tmp_path, counter)
    started = datetime(2026, 3, 2, 10, 0, tzinfo=manager.tz)
    manager.now = lambda: started  # type: ignore[method-assign]

    with patch(_NOTIFY, side_effect=RuntimeError("webhook down")):
        assert manager.run_job("daily_auto_order", reason="scheduled") is False
    assert counter.get("daily_auto_order", 0) == 0
