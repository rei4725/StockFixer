"""補完のポーリングで1つのジョブが落ちても後のジョブを補完すること（#1186）"""

import logging
import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

_CONFIG = {
    "first_job": {"description": "先頭"},
    "today_job": {"description": "当日分"},
    "backfill_job": {"description": "前日分"},
}


def _queue_manager(fail_on: str) -> MagicMock:
    """先頭のジョブだけが例外を出す queue_manager を作る"""
    qm = MagicMock()
    qm.should_recover.side_effect = lambda job_id, now: job_id in ("first_job", "today_job")

    def _missed(job_id, lookback_days, now):
        return ["2026-10-25"] if job_id == "backfill_job" else []

    qm.get_missed_past_periods.side_effect = _missed

    def _run_job(job_id, reason, **kwargs):
        if fail_on == "run_job" and job_id == "first_job":
            raise RuntimeError("先頭のジョブが落ちた")

    qm.run_job.side_effect = _run_job
    if fail_on == "should_recover":
        original = qm.should_recover.side_effect

        def _should_recover(job_id, now):
            if job_id == "first_job":
                raise RuntimeError("先頭のジョブが落ちた")
            return original(job_id, now)

        qm.should_recover.side_effect = _should_recover
    return qm


def _poll(qm: MagicMock) -> None:
    import run_scheduler

    with (
        patch.object(run_scheduler, "SCHEDULE_CONFIG", _CONFIG),
        patch.object(run_scheduler, "ping_heartbeat"),
    ):
        run_scheduler._poll_recovery_jobs(qm)


def test_failed_job_does_not_skip_later_recoveries(caplog):
    """先頭のジョブの補完が例外で落ちても、後ろの当日分と前日分の補完が同じ回で動くこと"""
    qm = _queue_manager("run_job")

    with caplog.at_level(logging.ERROR):
        _poll(qm)

    qm.run_job.assert_any_call("today_job", reason="recovery")
    qm.run_job.assert_any_call("backfill_job", reason="recovery_backfill", period_key="2026-10-25")
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("first_job" in r.getMessage() and r.exc_info for r in errors)


def test_failed_should_recover_does_not_skip_later_recoveries(caplog):
    """先頭のジョブの補完の判定が例外で落ちても、後ろのジョブの補完が動くこと"""
    qm = _queue_manager("should_recover")

    with caplog.at_level(logging.ERROR):
        _poll(qm)

    qm.run_job.assert_any_call("today_job", reason="recovery")
    qm.run_job.assert_any_call("backfill_job", reason="recovery_backfill", period_key="2026-10-25")
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("first_job" in r.getMessage() and r.exc_info for r in errors)


def test_each_job_is_judged_with_time_measured_just_before():
    """前のジョブの長い補完で 9:30 を越えたら、同じ回の自動発注は越えた後の時刻で判定され補完されないこと"""
    clock = {"now": datetime(2026, 10, 27, 9, 5)}
    qm = MagicMock()
    qm.now.side_effect = lambda: clock["now"]
    seen: dict = {}

    def _should_recover(job_id, now):
        seen[job_id] = now
        if job_id == "daily_auto_order":
            return (now.hour, now.minute) < (9, 30)
        return True

    def _run_job(job_id, reason, **kwargs):
        # 前のジョブの補完の間に 9:30 を越える
        if job_id == "daily_pipeline":
            clock["now"] = datetime(2026, 10, 27, 9, 35)

    qm.should_recover.side_effect = _should_recover
    qm.get_missed_past_periods.return_value = []
    qm.run_job.side_effect = _run_job
    config = {
        "daily_pipeline": {"description": "日次の道"},
        "daily_auto_order": {"description": "自動発注"},
    }

    import run_scheduler

    with (
        patch.object(run_scheduler, "SCHEDULE_CONFIG", config),
        patch.object(run_scheduler, "ping_heartbeat"),
    ):
        run_scheduler._poll_recovery_jobs(qm)

    assert seen["daily_pipeline"] == datetime(2026, 10, 27, 9, 5)
    assert seen["daily_auto_order"] == datetime(2026, 10, 27, 9, 35)
    assert [c.args[0] for c in qm.run_job.call_args_list] == ["daily_pipeline"]
