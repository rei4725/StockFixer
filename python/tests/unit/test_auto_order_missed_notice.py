"""8:50 の自動発注の回が消えた日・締めまで動かなかった日の知らせ（#1674）"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
import run_scheduler

from src.orchestration.scheduler_queue import SchedulerQueueManager

_JST = ZoneInfo("Asia/Tokyo")
_SEND = "src.reporting.discord.discord_utils.send_webhook_text"
_JOB = "daily_auto_order"
_DAY = "2026-10-28"  # 本番で回が消えた水曜


def _at(hour: int, minute: int = 0, day: int = 28, month: int = 10) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=_JST)


def _started(period_key: str = _DAY, boot_id: str = "old-boot") -> dict:
    return {
        "job_id": _JOB,
        "reason": "scheduled",
        "status": "started",
        "period_key": period_key,
        "run_id": "run-1",
        "boot_id": boot_id,
        "started_at": "2026-10-27T23:50:00+00:00",
    }


def _finished(status: str, period_key: str = _DAY) -> dict:
    return {
        "job_id": _JOB,
        "reason": "scheduled",
        "status": status,
        "period_key": period_key,
        "run_id": "run-1",
        "started_at": "2026-10-27T23:50:00+00:00",
        "finished_at": "2026-10-27T23:51:00+00:00",
        "duration_seconds": 60.0,
        "error": None,
    }


def _manager(tmp_path: Path, events: list[dict], *, write: bool = True) -> SchedulerQueueManager:
    """本番の欄（締め・上限を含む）のまま、関数だけ差し替える。"""
    config = dict(run_scheduler.SCHEDULE_CONFIG[_JOB])
    config["func"] = MagicMock()
    state_path = tmp_path / "state.json"
    if write:
        state_path.write_text(json.dumps({"events": events}), encoding="utf-8")
    return SchedulerQueueManager({_JOB: config}, state_file_path=str(state_path))


def _sent_texts(manager: SchedulerQueueManager, *times: datetime) -> list[str]:
    with patch(_SEND, return_value=True) as send:
        for now in times:
            run_scheduler._recover_job(manager, _JOB, now)
    return [call.args[0] for call in send.call_args_list]


class TestNotifies:
    @pytest.mark.parametrize("now", [_at(9, 0), _at(9, 29), _at(9, 30), _at(15, 0)])
    def test_lost_run_notifies_once(self, tmp_path, now):
        manager = _manager(tmp_path, [_started()])
        texts = _sent_texts(manager, now, now + timedelta(minutes=1))

        assert len(texts) == 1
        assert _DAY in texts[0]
        assert "開始 1 回・失敗 0 回" in texts[0]
        manager.schedule_config[_JOB]["func"].assert_not_called()

    def test_failed_run_notifies_once(self, tmp_path):
        manager = _manager(tmp_path, [_started(), _finished("error")])
        texts = _sent_texts(manager, _at(9, 5), _at(9, 40))

        assert len(texts) == 1
        assert "開始 1 回・失敗 1 回" in texts[0]

    @pytest.mark.parametrize("now", [_at(9, 30), _at(23, 59)])
    def test_no_run_past_the_cutoff_notifies_once(self, tmp_path, now):
        manager = _manager(tmp_path, [])
        texts = _sent_texts(manager, now, now + timedelta(seconds=30))

        assert len(texts) == 1
        assert _DAY in texts[0]
        assert "開始 0 回・失敗 0 回" in texts[0]
        manager.schedule_config[_JOB]["func"].assert_not_called()

    @pytest.mark.parametrize("events", [[_started()], []], ids=["lost_run", "no_run"])
    def test_no_second_notice_after_restart(self, tmp_path, events):
        manager = _manager(tmp_path, events)
        first = _sent_texts(manager, _at(9, 40))
        # 再起動の後の新しい起動は、状態ファイルだけを引き継ぐ
        restarted = _manager(tmp_path, [], write=False)
        second = _sent_texts(restarted, _at(10, 0), _at(16, 0))

        assert len(first) == 1
        assert second == []

    def test_next_business_day_is_a_new_period(self, tmp_path):
        manager = _manager(tmp_path, [])
        texts = _sent_texts(manager, _at(9, 40), _at(9, 40, day=29))

        assert len(texts) == 2
        assert "2026-10-29" in texts[1]

    @pytest.mark.parametrize("events", [[_started()], []], ids=["lost_run", "no_run"])
    def test_message_says_what_may_be_missing(self, tmp_path, events):
        manager = _manager(tmp_path, events)
        text = _sent_texts(manager, _at(9, 40))[0]

        assert "自動発注" in text
        assert "やり直さない" in text
        for word in ("SL/TP", "売りシグナル", "新規買い"):
            assert word in text
        assert "途中までの注文" in text

    def test_failed_send_is_retried_at_the_next_poll(self, tmp_path):
        manager = _manager(tmp_path, [])
        with patch(_SEND, side_effect=[False, True]) as send:
            run_scheduler._recover_job(manager, _JOB, _at(9, 40))
            run_scheduler._recover_job(manager, _JOB, _at(9, 41))
            run_scheduler._recover_job(manager, _JOB, _at(9, 42))

        assert send.call_count == 2


class TestNoNotice:
    @pytest.mark.parametrize("now", [_at(9, 5), _at(9, 40)])
    def test_day_with_success(self, tmp_path, now):
        manager = _manager(tmp_path, [_started(), _finished("success")])
        assert _sent_texts(manager, now) == []

    def test_before_the_scheduled_time(self, tmp_path):
        # 前の日の回の記録があっても、今日の定時より前には何も言わない
        manager = _manager(tmp_path, [_started(period_key="2026-10-27")])
        assert _sent_texts(manager, _at(8, 30)) == []

    def test_lost_run_before_the_recovery_delay(self, tmp_path):
        manager = _manager(tmp_path, [_started()])
        assert _sent_texts(manager, _at(8, 59)) == []

    def test_run_in_progress_in_this_process(self, tmp_path):
        manager = _manager(tmp_path, [_started(boot_id="this-boot")])
        manager._in_progress_jobs.add(_JOB)
        assert _sent_texts(manager, _at(9, 5), _at(9, 40)) == []

    def test_no_run_before_the_cutoff_is_left_to_the_recovery(self, tmp_path):
        manager = _manager(tmp_path, [])
        assert _sent_texts(manager, _at(9, 29)) == []

        manager.schedule_config[_JOB]["func"].assert_called_once()

    @pytest.mark.parametrize("day", [31, 25], ids=["saturday", "sunday"])
    def test_weekend(self, tmp_path, day):
        manager = _manager(tmp_path, [])
        assert _sent_texts(manager, _at(10, 0, day=day)) == []

    @pytest.mark.parametrize("events", [[_started("2026-11-03")], []], ids=["lost_run", "no_run"])
    def test_weekday_market_holiday(self, tmp_path, events):
        # 2026-11-03 は文化の日の火曜
        manager = _manager(tmp_path, events)
        assert _sent_texts(manager, _at(10, 0, day=3, month=11)) == []

    def test_other_job_with_a_cutoff(self, tmp_path):
        job_id = "daily_horizon_exit"
        config = dict(run_scheduler.SCHEDULE_CONFIG[job_id])
        config["func"] = MagicMock()
        manager = SchedulerQueueManager(
            {job_id: config}, state_file_path=str(tmp_path / "state.json")
        )
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, job_id, _at(9, 40))

        send.assert_not_called()


class TestRecoveryJudgementIsUnchanged:
    def test_lost_run_is_not_recovered_and_the_notice_is_not_counted(self, tmp_path):
        manager = _manager(tmp_path, [_started()])
        now = _at(9, 10)
        count_before = manager.get_execution_count(_JOB, _DAY)
        assert manager.should_recover(_JOB, now=now) is False

        assert len(_sent_texts(manager, now)) == 1

        assert manager.get_execution_count(_JOB, _DAY) == count_before
        assert manager.should_recover(_JOB, now=now) is False
        assert manager.has_success(_JOB, _DAY) is False
        manager.schedule_config[_JOB]["func"].assert_not_called()

    def test_past_the_cutoff_the_notice_does_not_change_the_count(self, tmp_path):
        manager = _manager(tmp_path, [])
        assert len(_sent_texts(manager, _at(9, 40))) == 1

        assert manager.get_execution_count(_JOB, _DAY) == 0
        assert manager.should_recover(_JOB, now=_at(9, 40)) is False

    def test_schedule_fields_are_unchanged(self):
        config = run_scheduler.SCHEDULE_CONFIG[_JOB]
        assert (config["hour"], config["minute"]) == (8, 50)
        assert config["recovery_until"] == (9, 30)
        assert config["max_executions_per_period"] == 1
        assert config["max_retries"] == 0
        assert not config.get("rerun_if_interrupted", False)
