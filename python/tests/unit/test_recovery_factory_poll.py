"""戦略ファクトリーの補完が、日次の道の補完をふさがず、週の最適化の補完にもふさがれないこと（#1957）"""

import os
import sys
import threading
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

_WAIT = 5
_FACTORY = "nightly_strategy_factory"
_PAST_PERIOD = "2026-11-16"


def _interval_jobs(config, queue_manager) -> dict:
    """_register_jobs が登録した間隔のジョブの関数を、ジョブの id で返す"""
    import run_scheduler

    scheduler = MagicMock()
    with patch.object(run_scheduler, "SCHEDULE_CONFIG", config):
        run_scheduler._register_jobs(scheduler, queue_manager)
    return {
        c.kwargs["id"]: c.args[0]
        for c in scheduler.add_job.call_args_list
        if c.kwargs.get("trigger") == "interval"
    }


def _tick(jobs: dict, threads: dict) -> None:
    """動いていない間隔のジョブだけを起こす（max_instances=1）"""
    for job_id, func in jobs.items():
        if job_id in threads and threads[job_id].is_alive():
            continue
        threads[job_id] = threading.Thread(target=func, daemon=True)
        threads[job_id].start()
        threads[job_id].join(timeout=0.2)


def _production_config(*job_ids) -> dict:
    import run_scheduler

    return {job_id: dict(run_scheduler.SCHEDULE_CONFIG[job_id]) for job_id in job_ids}


def test_factory_is_not_judged_by_recovery_poller():
    """本番の設定で、戦略ファクトリーの当日分と前日分の補完を recovery_poller が判定しないこと"""
    import run_scheduler

    qm = MagicMock()
    qm.should_recover.return_value = False
    qm.get_missed_past_periods.return_value = []
    jobs = _interval_jobs(run_scheduler.SCHEDULE_CONFIG, qm)

    judged: dict = {}
    with patch.object(run_scheduler, "ping_heartbeat"):
        for poller_id, func in jobs.items():
            qm.should_recover.reset_mock()
            qm.get_missed_past_periods.reset_mock()
            func()
            judged[poller_id] = {c.args[0] for c in qm.should_recover.call_args_list} | {
                c.args[0] for c in qm.get_missed_past_periods.call_args_list
            }

    assert _FACTORY not in judged["recovery_poller"]
    assert _FACTORY not in judged["recovery_long_job_poller"]
    assert [p for p, ids in judged.items() if _FACTORY in ids] == ["recovery_side_job_poller"]


def test_daily_pipeline_recovered_while_factory_recovery_runs():
    """本番の設定で、戦略ファクトリーの補完が戻らない間に補完の時刻を迎えた日次の道が補完されること"""
    import run_scheduler

    config = _production_config("daily_pipeline", _FACTORY)
    release = threading.Event()
    started = threading.Event()
    ran: list = []

    def _run_job(job_id, reason, **kwargs):
        ran.append(job_id)
        if job_id == _FACTORY:
            started.set()
            release.wait(timeout=_WAIT)

    def _should_recover(job_id, now):
        # 日次の道は、戦略ファクトリーの補完が走り出した後に補完の時刻を迎える
        if job_id == "daily_pipeline" and not started.is_set():
            return False
        return job_id not in ran

    qm = MagicMock()
    qm.should_recover.side_effect = _should_recover
    qm.get_missed_past_periods.return_value = []
    qm.run_job.side_effect = _run_job
    jobs = _interval_jobs(config, qm)
    threads: dict = {}

    with (
        patch.object(run_scheduler, "SCHEDULE_CONFIG", config),
        patch.object(run_scheduler, "ping_heartbeat"),
    ):
        try:
            _tick(jobs, threads)
            assert started.wait(timeout=_WAIT)
            _tick(jobs, threads)
            ran_while_blocked = list(ran)
        finally:
            release.set()
            for t in threads.values():
                t.join(timeout=_WAIT)

    assert ran_while_blocked == [_FACTORY, "daily_pipeline"]


def test_factory_backfill_runs_while_weekly_optimization_recovery_blocks():
    """本番の設定で、週の最適化の補完が戻らない間も、戦略ファクトリーの前日分の補完が走ること"""
    import run_scheduler

    config = _production_config("weekly_optimization", "weekly_walk_forward_report", _FACTORY)
    release = threading.Event()
    started = threading.Event()
    calls: list = []

    def _run_job(job_id, reason, **kwargs):
        calls.append((job_id, reason, kwargs.get("period_key")))
        if job_id == "weekly_optimization":
            started.set()
            # 約29時間の補完は戻らない（テストが止まり続けないよう上限を置く）
            release.wait(timeout=_WAIT)

    qm = MagicMock()
    qm.should_recover.side_effect = lambda job_id, now: job_id == "weekly_optimization" and not (
        started.is_set()
    )
    qm.get_missed_past_periods.side_effect = lambda job_id, lookback_days, now: (
        [_PAST_PERIOD] if job_id == _FACTORY else []
    )
    qm.run_job.side_effect = _run_job
    jobs = _interval_jobs(config, qm)
    threads: dict = {}

    with (
        patch.object(run_scheduler, "SCHEDULE_CONFIG", config),
        patch.object(run_scheduler, "ping_heartbeat"),
    ):
        try:
            _tick(jobs, threads)
            assert started.wait(timeout=_WAIT)
            _tick(jobs, threads)
            calls_while_blocked = list(calls)
        finally:
            release.set()
            for t in threads.values():
                t.join(timeout=_WAIT)

    assert ("weekly_optimization", "recovery", None) in calls_while_blocked
    assert (_FACTORY, "recovery_backfill", _PAST_PERIOD) in calls_while_blocked
