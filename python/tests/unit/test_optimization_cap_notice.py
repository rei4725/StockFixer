"""週の最適化が回数の上限で補完をやめた週の知らせ（#1647）"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
import run_scheduler

from src.backtest.optimizer.persistence import OptimalParamsCorruptedError
from src.orchestration.scheduler_queue import SchedulerQueueManager

_JOB_ID = "weekly_optimization"
_NOW = datetime(2026, 9, 3, 10, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
_PERIOD = "2026-W36"
_NEXT_WEEK_NOW = _NOW + timedelta(days=7)
_NEXT_PERIOD = "2026-W37"
_SEND = "src.reporting.discord.discord_utils.send_webhook_text"
_TASKS = "src.watchlist.batch_runner.load_target_symbols"
_PARAMS = "src.backtest.optimizer.batch.get_optimal_params"


def _config() -> dict:
    """本番の欄（上限の 40 を含む）のまま、関数だけ差し替える。"""
    config = dict(run_scheduler.SCHEDULE_CONFIG[_JOB_ID])
    config["func"] = MagicMock()
    return {_JOB_ID: config}


def _interrupted(count: int, period_key: str = _PERIOD) -> list[dict]:
    return [
        {
            "job_id": _JOB_ID,
            "reason": "recovery",
            "status": "started",
            "period_key": period_key,
            "run_id": f"{period_key}-run-{index}",
            "boot_id": f"old-boot-{index}",
            "started_at": "2026-09-02T00:00:00+00:00",
        }
        for index in range(count)
    ]


def _finished(status: str, run_id: str, period_key: str = _PERIOD) -> dict:
    return {
        "job_id": _JOB_ID,
        "reason": "recovery",
        "status": status,
        "period_key": period_key,
        "run_id": run_id,
        "started_at": "2026-09-02T00:00:00+00:00",
        "finished_at": "2026-09-02T01:00:00+00:00",
        "duration_seconds": 3600.0,
        "error": None,
    }


def _manager(tmp_path: Path, events: list[dict], *, write: bool = True) -> SchedulerQueueManager:
    state_path = tmp_path / "state.json"
    if write:
        state_path.write_text(json.dumps({"events": events}), encoding="utf-8")
    return SchedulerQueueManager(_config(), state_file_path=str(state_path))


def _params_for(symbol_days: dict[str, int | None]):
    """銘柄ごとに、最後の最適化が何日前かを返す get_optimal_params の差し替えを作る。"""

    def fake(_market: str, symbol: str):
        days = symbol_days[symbol]
        if days is None:
            return None
        return {"timestamp": (datetime.now() - timedelta(days=days)).isoformat()}

    return fake


@pytest.fixture
def backlog():
    """3銘柄のうち2銘柄が6日より前に最適化した銘柄である既定の見込み。"""
    tasks = [{"market": "jp", "symbol": symbol} for symbol in ("1", "2", "3")]
    with (
        patch(_TASKS, return_value=tasks),
        patch(_PARAMS, side_effect=_params_for({"1": 1, "2": 10, "3": None})),
    ):
        yield


@pytest.mark.usefixtures("backlog")
class TestCapReached:
    def test_notifies_once_with_remaining_when_capped_without_success(self, tmp_path):
        # 消えた回を 40 回まで許し、残りの 2 回で実行の上限 2 に届く
        manager = _manager(tmp_path, _interrupted(42))
        assert manager.should_recover(_JOB_ID, now=_NOW) is False

        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, _JOB_ID, _NOW)
            run_scheduler._recover_job(manager, _JOB_ID, _NOW + timedelta(minutes=5))

        send.assert_called_once()
        text = send.call_args.args[0]
        assert _PERIOD in text
        assert "2 / 3 銘柄" in text
        assert "開始 42 回" in text
        assert "SL/TP" in text

    def test_restart_does_not_send_again(self, tmp_path):
        manager = _manager(tmp_path, _interrupted(42))
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, _JOB_ID, _NOW)
            # 再起動・デプロイの後の新しい起動は、状態ファイルだけを引き継ぐ
            restarted = _manager(tmp_path, [], write=False)
            run_scheduler._recover_job(restarted, _JOB_ID, _NOW + timedelta(hours=1))

        send.assert_called_once()

    def test_next_week_notifies_again(self, tmp_path):
        events = _interrupted(42) + _interrupted(42, _NEXT_PERIOD)
        manager = _manager(tmp_path, events)
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, _JOB_ID, _NOW)
            run_scheduler._recover_job(manager, _JOB_ID, _NEXT_WEEK_NOW)

        assert send.call_count == 2
        assert _NEXT_PERIOD in send.call_args.args[0]

    def test_two_failed_runs_are_also_a_capped_week(self, tmp_path):
        events = [_finished("error", "run-a"), _finished("error", "run-b")]
        manager = _manager(tmp_path, events)
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, _JOB_ID, _NOW)

        send.assert_called_once()
        assert "失敗 2 回" in send.call_args.args[0]

    def test_notice_record_does_not_change_the_recovery_judgement(self, tmp_path):
        manager = _manager(tmp_path, _interrupted(42))
        count_before = manager.get_execution_count(_JOB_ID, _PERIOD)
        with patch(_SEND, return_value=True):
            run_scheduler._recover_job(manager, _JOB_ID, _NOW)

        assert manager.get_execution_count(_JOB_ID, _PERIOD) == count_before
        assert manager.should_recover(_JOB_ID, now=_NOW) is False
        assert manager.has_success(_JOB_ID, _PERIOD) is False


@pytest.mark.usefixtures("backlog")
class TestNoNotice:
    def test_week_with_success(self, tmp_path):
        manager = _manager(tmp_path, _interrupted(42) + [_finished("success", "run-ok")])
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, _JOB_ID, _NOW)

        send.assert_not_called()

    @pytest.mark.parametrize("interrupted", [0, 1, 40, 41])
    def test_week_below_the_cap(self, tmp_path, interrupted):
        manager = _manager(tmp_path, _interrupted(interrupted))
        with (
            patch.object(manager, "should_recover", return_value=False),
            patch(_SEND, return_value=True) as send,
        ):
            run_scheduler._recover_job(manager, _JOB_ID, _NOW)

        send.assert_not_called()

    def test_run_in_progress(self, tmp_path):
        manager = _manager(tmp_path, _interrupted(42))
        manager._in_progress_jobs.add(_JOB_ID)
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, _JOB_ID, _NOW)

        send.assert_not_called()

    def test_before_the_scheduled_time_of_the_week(self, tmp_path):
        # 同じ週の水曜の定時より前の手での回で上限に届いていても、補完の判定の前は知らせない
        monday = _NOW - timedelta(days=3)
        manager = _manager(tmp_path, _interrupted(42))
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, _JOB_ID, monday)

        send.assert_not_called()

    def test_other_jobs_are_not_covered(self, tmp_path):
        state_path = tmp_path / "state.json"
        config = {
            "weekly_report": {
                "func": MagicMock(),
                "period": "weekly",
                "day_of_week": "thu",
                "hour": 2,
                "minute": 0,
                "max_executions_per_period": 1,
            }
        }
        event = {**_finished("error", "run-a"), "job_id": "weekly_report"}
        state_path.write_text(json.dumps({"events": [event]}), encoding="utf-8")
        manager = SchedulerQueueManager(config, state_file_path=str(state_path))
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, "weekly_report", _NOW)

        send.assert_not_called()


@pytest.mark.usefixtures("backlog")
class TestSendFailure:
    def test_raising_send_does_not_stop_the_polling(self, tmp_path):
        manager = _manager(tmp_path, _interrupted(42))
        with (
            patch(_SEND, side_effect=RuntimeError("discord down")),
            patch.object(
                manager, "get_missed_past_periods", wraps=manager.get_missed_past_periods
            ) as missed,
        ):
            run_scheduler._recover_job(manager, _JOB_ID, _NOW)

        missed.assert_called_once()

    def test_polling_moves_on_to_the_next_job(self, tmp_path):
        manager = _manager(tmp_path, _interrupted(42))
        with (
            patch(_SEND, side_effect=RuntimeError("discord down")),
            patch.object(manager, "now", return_value=_NOW),
            patch.object(run_scheduler, "logger") as logger,
        ):
            run_scheduler._poll_recovery_jobs(manager, long_recovery=True)

        logger.error.assert_not_called()

    @pytest.mark.parametrize("failure", [{"return_value": False}, {"side_effect": RuntimeError()}])
    def test_failed_send_is_retried_by_the_next_poll(self, tmp_path, failure):
        manager = _manager(tmp_path, _interrupted(42))
        with patch(_SEND, **failure):
            run_scheduler._recover_job(manager, _JOB_ID, _NOW)
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, _JOB_ID, _NOW + timedelta(minutes=5))
            run_scheduler._recover_job(manager, _JOB_ID, _NOW + timedelta(minutes=10))

        send.assert_called_once()

    def test_failed_record_does_not_repeat_the_send_in_the_same_process(self, tmp_path):
        manager = _manager(tmp_path, _interrupted(42))
        with (
            patch(_SEND, return_value=True) as send,
            patch.object(manager, "_append_event", side_effect=OSError("disk full")),
        ):
            run_scheduler._recover_job(manager, _JOB_ID, _NOW)
            run_scheduler._recover_job(manager, _JOB_ID, _NOW + timedelta(minutes=5))

        send.assert_called_once()


class TestBacklogUnknown:
    def test_notice_still_goes_out_when_backlog_cannot_be_read(self, tmp_path):
        manager = _manager(tmp_path, _interrupted(42))
        with (
            patch(_TASKS, return_value=[{"market": "jp", "symbol": "1"}]),
            patch(_PARAMS, side_effect=OptimalParamsCorruptedError("broken")),
            patch(_SEND, return_value=True) as send,
        ):
            run_scheduler._recover_job(manager, _JOB_ID, _NOW)

        send.assert_called_once()
        assert "数えられなかった" in send.call_args.args[0]


class TestCountOptimizationBacklog:
    @staticmethod
    def _count(tasks: list[dict]) -> tuple[int, date | None]:
        from src.backtest.optimizer.batch import count_optimization_backlog

        return count_optimization_backlog(tasks, skip_days=6)

    def test_counts_symbols_the_batch_would_not_skip(self):
        tasks = [{"market": "jp", "symbol": symbol} for symbol in ("new", "old", "none")]
        fake = _params_for({"new": 5, "old": 8, "none": None})
        with patch(_PARAMS, side_effect=fake):
            remaining, oldest = self._count(tasks)

        assert remaining == 2
        assert oldest == (datetime.now() - timedelta(days=8)).date()

    def test_nothing_remaining(self):
        tasks = [{"market": "jp", "symbol": "new"}]
        with patch(_PARAMS, side_effect=_params_for({"new": 1})):
            assert self._count(tasks) == (0, None)

    def test_symbols_without_a_record_have_no_oldest_date(self):
        tasks = [{"market": "jp", "symbol": "none"}]
        with patch(_PARAMS, side_effect=_params_for({"none": None})):
            remaining, oldest = self._count(tasks)

        assert remaining == 1
        assert oldest is None
