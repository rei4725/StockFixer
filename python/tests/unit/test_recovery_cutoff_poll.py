"""締めを持つジョブの補完を、長い補完でふさがるポーリングと別のジョブで判定すること（#1410）"""

import os
import sys
import threading
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.orchestration.scheduler_queue import SchedulerQueueManager  # noqa: E402

_WAIT = 5


def _interval_pollers(config, queue_manager) -> list:
    """_register_jobs が登録した間隔のジョブ（補完のポーリング）の関数を返す"""
    import run_scheduler

    scheduler = MagicMock()
    with patch.object(run_scheduler, "SCHEDULE_CONFIG", config):
        run_scheduler._register_jobs(scheduler, queue_manager)
    return [
        c.args[0] for c in scheduler.add_job.call_args_list if c.kwargs.get("trigger") == "interval"
    ]


def test_cutoff_job_recovered_while_long_recovery_blocks_poll():
    """日次の道の補完がふさがっている間に 9:00 を迎えても、自動発注の当日分を 9:30 より前に補完すること"""
    import run_scheduler

    cron = {"trigger": "cron", "day_of_week": "mon-fri"}
    config = {
        "daily_pipeline": {**cron, "hour": 7, "minute": 30, "description": "日次の道"},
        "daily_auto_order": {
            **cron,
            "hour": 8,
            "minute": 50,
            "recovery_until": (9, 30),
            "description": "自動発注",
        },
    }
    qm = MagicMock()
    pollers = _interval_pollers(config, qm)
    clock = {"now": datetime(2026, 10, 27, 8, 55)}
    release = threading.Event()
    pipeline_started = threading.Event()
    ran: dict = {}
    qm.now.side_effect = lambda: clock["now"]

    def _should_recover(job_id, now):
        if job_id in ran:
            return False
        if job_id == "daily_auto_order":
            return (9, 0) <= (now.hour, now.minute) < (9, 30)
        return True

    def _run_job(job_id, reason, **kwargs):
        ran[job_id] = clock["now"]
        if job_id == "daily_pipeline":
            pipeline_started.set()
            # 日次の道の補完は終わらずにポーリングをふさぐ（テストが止まり続けないよう上限を置く）
            release.wait(timeout=_WAIT)

    qm.should_recover.side_effect = _should_recover
    qm.get_missed_past_periods.return_value = []
    qm.run_job.side_effect = _run_job

    threads: dict = {}
    with (
        patch.object(run_scheduler, "SCHEDULE_CONFIG", config),
        patch.object(run_scheduler, "ping_heartbeat"),
    ):
        try:
            # 08:55 から 09:35 まで5分おきに、動いていないポーリングだけを起こす（max_instances=1）
            for step in range(9):
                clock["now"] = datetime(2026, 10, 27, 8, 55) + timedelta(minutes=5 * step)
                for i, poller in enumerate(pollers):
                    if i in threads and threads[i].is_alive():
                        continue
                    threads[i] = threading.Thread(target=poller, args=(), daemon=True)
                    threads[i].start()
                    threads[i].join(timeout=0.2)
                if step == 0:
                    assert pipeline_started.wait(timeout=_WAIT)
        finally:
            release.set()
            for t in threads.values():
                t.join(timeout=_WAIT)

    assert "daily_auto_order" in ran
    assert ran["daily_auto_order"] < datetime(2026, 10, 27, 9, 30)


def test_each_job_is_judged_by_one_poller_only():
    """締めのあるジョブは締めのポーリングだけ、締めの無いジョブは今のポーリングだけが判定すること"""
    import run_scheduler

    config = {
        "daily_pipeline": {"description": "日次の道"},
        "daily_auto_order": {"description": "自動発注", "recovery_until": (9, 30)},
        "daily_report": {"description": "日次の報告"},
    }
    qm = MagicMock()
    qm.should_recover.return_value = False
    qm.get_missed_past_periods.return_value = []

    with (
        patch.object(run_scheduler, "SCHEDULE_CONFIG", config),
        patch.object(run_scheduler, "ping_heartbeat"),
    ):
        run_scheduler._poll_recovery_jobs(qm, with_cutoff=True)
        cutoff_seen = [c.args[0] for c in qm.should_recover.call_args_list]
        qm.should_recover.reset_mock()
        run_scheduler._poll_recovery_jobs(qm, with_cutoff=False)
        plain_seen = [c.args[0] for c in qm.should_recover.call_args_list]

    assert cutoff_seen == ["daily_auto_order"]
    assert plain_seen == ["daily_pipeline", "daily_report"]


def test_same_job_does_not_run_twice_from_two_pollers(tmp_path):
    """締めのジョブの補完が動いている間・済んだ後に両方のポーリングが回っても、1回しか走らないこと"""
    import run_scheduler

    started = threading.Event()
    release = threading.Event()
    calls = {"n": 0}

    def _slow_order():
        calls["n"] += 1
        started.set()
        release.wait(timeout=_WAIT)

    config = {
        "daily_auto_order": {
            "func": _slow_order,
            "trigger": "cron",
            "period": "daily",
            "day_of_week": "mon-fri",
            "hour": 8,
            "minute": 50,
            "recovery_delay_minutes": 10,
            "recovery_until": (9, 30),
            "max_executions_per_period": 1,
            "max_retries": 0,
            "description": "自動発注",
        },
    }
    manager = SchedulerQueueManager(config, state_file_path=str(tmp_path / "state.json"))
    # 2026-10-27 は火曜
    poll_now = datetime(2026, 10, 27, 9, 5, tzinfo=manager.tz)
    manager.now = lambda: poll_now  # type: ignore[method-assign]

    with (
        patch.object(run_scheduler, "SCHEDULE_CONFIG", config),
        patch.object(run_scheduler, "ping_heartbeat"),
        patch("src.orchestration.scheduler_queue.ping_heartbeat"),
    ):
        t = threading.Thread(
            target=run_scheduler._poll_recovery_jobs, args=(manager, True), daemon=True
        )
        t.start()
        try:
            assert started.wait(timeout=_WAIT)
            # 動いている間（_in_progress_jobs）
            run_scheduler._poll_recovery_jobs(manager, with_cutoff=True)
            run_scheduler._poll_recovery_jobs(manager, with_cutoff=False)
            assert calls["n"] == 1
        finally:
            release.set()
            t.join(timeout=_WAIT)
        # 済んだ後（回数の上限と成功の記録）
        run_scheduler._poll_recovery_jobs(manager, with_cutoff=True)
        run_scheduler._poll_recovery_jobs(manager, with_cutoff=False)
        assert manager.run_job("daily_auto_order", reason="recovery") is False

    assert calls["n"] == 1
