"""週の最適化の長い補完が、生存の ping と締めの無いほかのジョブの補完をふさがないこと（#1542）"""

import os
import sys
import threading
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.heartbeat import SCHEDULER_ALIVE_SLUG  # noqa: E402

_WAIT = 5
_CRON = {"trigger": "cron", "day_of_week": "mon-fri", "hour": 7, "minute": 30}


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


def _blocking_queue_manager(blocking_job_id: str, release: threading.Event):
    """blocking_job_id の補完が release まで戻らない queue_manager を作る"""
    qm = MagicMock()
    started = threading.Event()
    ran: list = []

    def _run_job(job_id, reason, **kwargs):
        ran.append(job_id)
        if job_id == blocking_job_id:
            started.set()
            # 長い補完は戻らない（テストが止まり続けないよう上限を置く）
            release.wait(timeout=_WAIT)

    qm.should_recover.side_effect = lambda job_id, now: job_id not in ran
    qm.get_missed_past_periods.return_value = []
    qm.run_job.side_effect = _run_job
    return qm, started, ran


def test_alive_ping_keeps_going_while_recovery_call_blocks():
    """補完の呼び出しが戻らない間も、生存の ping が別のジョブで送られ続けること"""
    import run_scheduler

    config = {"daily_pipeline": {**_CRON, "description": "日次の道"}}
    release = threading.Event()
    qm, started, _ = _blocking_queue_manager("daily_pipeline", release)
    jobs = _interval_jobs(config, qm)
    threads: dict = {}

    with (
        patch.object(run_scheduler, "SCHEDULE_CONFIG", config),
        patch.object(run_scheduler, "ping_heartbeat") as ping,
    ):
        try:
            _tick(jobs, threads)
            assert started.wait(timeout=_WAIT)
            ping.reset_mock()
            # 補完が戻らないまま、次の間隔が来る
            _tick(jobs, threads)
            pings = [c.args[0] for c in ping.call_args_list]
        finally:
            release.set()
            for t in threads.values():
                t.join(timeout=_WAIT)

    assert pings == [SCHEDULER_ALIVE_SLUG]


def test_recovery_polls_do_not_send_alive_ping():
    """生存の ping は専用のジョブだけが送り、補完のポーリングは送らないこと"""
    import run_scheduler

    config = {"daily_pipeline": {"description": "日次の道"}}
    qm = MagicMock()
    qm.should_recover.return_value = False
    qm.get_missed_past_periods.return_value = []

    with (
        patch.object(run_scheduler, "SCHEDULE_CONFIG", config),
        patch.object(run_scheduler, "ping_heartbeat") as ping,
    ):
        run_scheduler._poll_recovery_jobs(qm)
        run_scheduler._poll_recovery_jobs(qm, with_cutoff=False)

    ping.assert_not_called()


def test_plain_job_recovered_while_weekly_optimization_recovery_runs():
    """weekly_optimization の補完が走っている間に、締めの無いほかのジョブが別のポーリングで補完されること"""
    import run_scheduler

    config = {
        "weekly_optimization": {**_CRON, "description": "週の最適化", "long_recovery": True},
        "daily_pipeline": {**_CRON, "description": "日次の道"},
    }
    release = threading.Event()
    qm, started, ran = _blocking_queue_manager("weekly_optimization", release)
    original = qm.should_recover.side_effect

    # 日次の道は、週の最適化の補完が走り出した後に補完の時刻を迎える
    def _should_recover(job_id, now):
        if job_id == "daily_pipeline" and not started.is_set():
            return False
        return original(job_id, now)

    qm.should_recover.side_effect = _should_recover
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

    assert ran_while_blocked == ["weekly_optimization", "daily_pipeline"]


def test_each_scheduled_job_is_judged_by_one_poller_only():
    """本番の設定の自動のジョブは、どれも1つのポーリングだけが判定し、週の最適化は専用のポーリングで判定すること"""
    import run_scheduler

    qm = MagicMock()
    qm.should_recover.return_value = False
    qm.get_missed_past_periods.return_value = []
    jobs = _interval_jobs(run_scheduler.SCHEDULE_CONFIG, qm)
    judged: dict = {}

    with patch.object(run_scheduler, "ping_heartbeat"):
        for poller_id, func in jobs.items():
            qm.should_recover.reset_mock()
            func()
            judged[poller_id] = [c.args[0] for c in qm.should_recover.call_args_list]

    seen = [job_id for job_ids in judged.values() for job_id in job_ids]
    expected = [
        job_id
        for job_id, config in run_scheduler.SCHEDULE_CONFIG.items()
        if config.get("auto_schedule", True)
    ]
    assert sorted(seen) == sorted(expected)
    assert ["weekly_optimization"] in judged.values()
    assert "weekly_optimization" not in judged["recovery_poller"]


def test_cutoff_job_is_not_judged_by_long_poller():
    """締めのあるジョブに長い補完の印があっても、締めのポーリングだけが判定すること"""
    import run_scheduler

    config = {
        "daily_auto_order": {
            **_CRON,
            "description": "自動発注",
            "recovery_until": (9, 30),
            "long_recovery": True,
        },
    }
    qm = MagicMock()
    qm.should_recover.return_value = False
    qm.get_missed_past_periods.return_value = []
    jobs = _interval_jobs(config, qm)

    judged = {}
    with (
        patch.object(run_scheduler, "SCHEDULE_CONFIG", config),
        patch.object(run_scheduler, "ping_heartbeat"),
    ):
        for poller_id, func in jobs.items():
            qm.should_recover.reset_mock()
            func()
            if qm.should_recover.called:
                judged[poller_id] = True

    assert list(judged) == ["recovery_cutoff_poller"]
