"""止まったまま日をまたいだ営業日の知らせ（#1679）"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
import run_scheduler

from src.orchestration.scheduler_cap_notice import CAP_NOTICE_STATUS, notify_recovery_cap
from src.orchestration.scheduler_queue import SchedulerQueueManager

_JST = ZoneInfo("Asia/Tokyo")
_SEND = "src.reporting.discord.discord_utils.send_webhook_text"
_JOBS = ["daily_auto_order", "daily_rule_signals"]
_AUTO, _RULE = _JOBS
_TUE, _WED, _THU, _FRI = "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23"


def _at(day: int, hour: int = 7, minute: int = 0, month: int = 10) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=_JST)


def _event(job_id: str, day: str, status: str, run_id: str = "run-1") -> dict[str, Any]:
    return {
        "job_id": job_id,
        "reason": "scheduled",
        "status": status,
        "period_key": day,
        "run_id": run_id,
        "boot_id": "old-boot",
        "started_at": f"{day}T07:30:00+00:00",
        "finished_at": f"{day}T07:31:00+00:00",
    }


def _ran(job_id: str, day: str, status: str = "success") -> list[dict[str, Any]]:
    return [_event(job_id, day, "started"), _event(job_id, day, status)]


# 見る範囲より古い記録。この日まで状態ファイルが保持していることを示す
_HISTORY = _ran(_AUTO, "2026-10-01") + _ran(_RULE, "2026-10-01")


def _fine_until(job_id: str, last_day: int) -> list[dict[str, Any]]:
    """見る範囲の頭の 10/16（金）から last_day まで、回が毎営業日成功した記録"""
    events = list(_HISTORY)
    for day in (16, *range(19, last_day + 1)):
        events += _ran(job_id, f"2026-10-{day}")
    return events


def _manager(
    tmp_path: Path, job_id: str, events: list[dict[str, Any]], *, write: bool = True, **extra: Any
) -> SchedulerQueueManager:
    """本番の欄のまま、関数だけ差し替える。"""
    config = {**run_scheduler.SCHEDULE_CONFIG[job_id], "func": MagicMock(), **extra}
    state_path = tmp_path / "state.json"
    if write:
        state_path.write_text(json.dumps({"events": events}), encoding="utf-8")
    return SchedulerQueueManager({job_id: config}, state_file_path=str(state_path))


def _sent_texts(manager: SchedulerQueueManager, job_id: str, *times: datetime) -> list[str]:
    with patch(_SEND, return_value=True) as send:
        for now in times:
            run_scheduler._recover_job(manager, job_id, now)
    return [call.args[0] for call in send.call_args_list]


@pytest.mark.parametrize("job_id", _JOBS)
class TestNotifies:
    def test_day_with_no_run_notifies_at_the_next_poll(self, tmp_path, job_id):
        # 木曜の回が1度も動かないまま、金曜の朝に起き直した
        manager = _manager(tmp_path, job_id, _fine_until(job_id, 21))
        texts = _sent_texts(manager, job_id, _at(23))

        assert len(texts) == 1
        assert _THU in texts[0]
        assert _WED not in texts[0]
        assert "動かなかった" in texts[0]
        manager.schedule_config[job_id]["func"].assert_not_called()

    def test_several_days_are_listed_in_one_notice(self, tmp_path, job_id):
        manager = _manager(tmp_path, job_id, _fine_until(job_id, 19))
        texts = _sent_texts(manager, job_id, _at(23))

        assert len(texts) == 1
        for day in (_TUE, _WED, _THU):
            assert day in texts[0]

    def test_restart_on_a_weekend_still_notifies(self, tmp_path, job_id):
        manager = _manager(tmp_path, job_id, _fine_until(job_id, 22))
        texts = _sent_texts(manager, job_id, _at(24, hour=12))

        assert len(texts) == 1
        assert _FRI in texts[0]

    def test_message_says_what_may_be_missing(self, tmp_path, job_id):
        manager = _manager(tmp_path, job_id, _fine_until(job_id, 21))
        text = _sent_texts(manager, job_id, _at(23))[0]

        assert "やり直さない" in text
        for word in ("SL/TP", "売りシグナル", "新規買い"):
            assert word in text

    def test_no_run_failed_and_lost_are_told_apart(self, tmp_path, job_id):
        events = _fine_until(job_id, 19)
        events += _ran(job_id, _TUE, "error")
        events += [_event(job_id, _WED, "started")]
        manager = _manager(tmp_path, job_id, events)
        text = _sent_texts(manager, job_id, _at(23))[0]

        lines = text.splitlines()
        assert any(_TUE in line and "失敗した" in line for line in lines)
        assert any(_WED in line and "途中で消えた" in line for line in lines)
        assert any(_THU in line and "動かなかった" in line for line in lines)

    def test_failed_send_is_retried_at_the_next_poll(self, tmp_path, job_id):
        manager = _manager(tmp_path, job_id, _fine_until(job_id, 21))
        with patch(_SEND, side_effect=[False, True]) as send:
            for minute in (0, 5, 10):
                run_scheduler._recover_job(manager, job_id, _at(23, minute=minute))

        assert send.call_count == 2


@pytest.mark.parametrize("job_id", _JOBS)
class TestOnlyOnce:
    def test_no_second_notice_in_the_same_boot(self, tmp_path, job_id):
        manager = _manager(tmp_path, job_id, _fine_until(job_id, 21))
        texts = _sent_texts(manager, job_id, _at(23), _at(23, minute=5))

        assert len(texts) == 1

    def test_no_second_notice_after_restart(self, tmp_path, job_id):
        manager = _manager(tmp_path, job_id, _fine_until(job_id, 21))
        first = _sent_texts(manager, job_id, _at(23))
        restarted = _manager(tmp_path, job_id, [], write=False)
        second = _sent_texts(restarted, job_id, _at(23, minute=5), _at(23, hour=8))

        assert len(first) == 1
        assert second == []

    def test_a_later_stop_notifies_only_the_new_day(self, tmp_path, job_id):
        manager = _manager(tmp_path, job_id, _fine_until(job_id, 20))
        first = _sent_texts(manager, job_id, _at(22))
        second = _sent_texts(manager, job_id, _at(23))

        assert _WED in first[0]
        assert _THU in second[0]
        assert _WED not in second[0]


class TestSameDayNoticeOfThePreviousDay:
    def test_no_second_notice_the_day_after(self, tmp_path):
        # 8:50 の回が抜けた水曜のうちに PR 1677 の知らせを送り、夜に止まって木曜に起きた
        events = _fine_until(_AUTO, 20) + _ran(_AUTO, "2026-10-14") + _ran(_AUTO, "2026-10-15")
        manager = _manager(tmp_path, _AUTO, events)
        same_day = _sent_texts(manager, _AUTO, _at(21, hour=9, minute=40))
        next_day = _sent_texts(manager, _AUTO, _at(22, hour=7))

        assert len(same_day) == 1
        assert next_day == []


@pytest.mark.parametrize("job_id", _JOBS)
class TestNoNotice:
    def test_days_with_success(self, tmp_path, job_id):
        manager = _manager(tmp_path, job_id, _fine_until(job_id, 22))
        assert _sent_texts(manager, job_id, _at(23)) == []

    def test_today_is_not_a_past_day(self, tmp_path, job_id):
        manager = _manager(tmp_path, job_id, _fine_until(job_id, 22))
        assert _sent_texts(manager, job_id, _at(23, hour=8)) == []

    def test_weekend_and_market_holiday(self, tmp_path, job_id):
        # 11/4（水）から見て 11/3 は文化の日、10/31 と 11/1 は土日で、回の記録が無い
        events = list(_HISTORY)
        for day in ("2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30", "2026-11-02"):
            events += _ran(job_id, day)
        manager = _manager(tmp_path, job_id, events)
        texts = _sent_texts(manager, job_id, _at(4, hour=7, month=11))

        assert texts == []

    def test_day_before_the_range(self, tmp_path, job_id):
        # 金曜の朝は前の週の金曜の 10/16 までを見る。動いていない 10/15 は範囲の外
        events = list(_HISTORY)
        for day in (16, 19, 20, 21, 22):
            events += _ran(job_id, f"2026-10-{day}")
        manager = _manager(tmp_path, job_id, events)
        assert _sent_texts(manager, job_id, _at(23)) == []

    def test_range_is_five_business_days(self, tmp_path, job_id):
        # 10/1 より後が1度も動いていなくても、16・19・20・21・22 の5日だけを並べる
        manager = _manager(tmp_path, job_id, list(_HISTORY))
        text = _sent_texts(manager, job_id, _at(23))[0]

        for day in ("2026-10-16", "2026-10-19", _TUE, _WED, _THU):
            assert day in text
        assert "2026-10-15" not in text

    def test_range_skips_a_market_holiday(self, tmp_path, job_id):
        # 11/4（水）から5営業日を数えると、文化の日と土日を飛ばして 10/27 までになる
        manager = _manager(tmp_path, job_id, list(_HISTORY))
        text = _sent_texts(manager, job_id, _at(4, hour=7, month=11))[0]

        for day in ("2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30", "2026-11-02"):
            assert day in text
        assert "2026-11-03" not in text
        assert "2026-10-26" not in text

    def test_days_the_state_file_no_longer_holds(self, tmp_path, job_id):
        # 記録が水曜の途中から始まる。それより前と、頭の切れた水曜そのものは読めない
        events = [_event("daily_settle_orders", _WED, "started")]
        manager = _manager(tmp_path, job_id, events)
        text = _sent_texts(manager, job_id, _at(23))[0]

        assert _THU in text
        for day in (_TUE, _WED):
            assert day not in text

    def test_state_with_no_events(self, tmp_path, job_id):
        manager = _manager(tmp_path, job_id, [])
        assert _sent_texts(manager, job_id, _at(23)) == []

    def test_broken_state_file(self, tmp_path, job_id):
        manager = _manager(tmp_path, job_id, _fine_until(job_id, 21))
        Path(manager.state_file_path).write_text("{broken", encoding="utf-8")
        assert _sent_texts(manager, job_id, _at(23)) == []

    def test_run_in_progress_in_this_process(self, tmp_path, job_id):
        # 日をまたいで動いている回の日を、抜けたと読まない
        manager = _manager(tmp_path, job_id, _fine_until(job_id, 21))
        manager._in_progress_jobs.add(job_id)
        assert _sent_texts(manager, job_id, _at(23, hour=0, minute=5)) == []


class TestStateFileIsReadOnceADay:
    def test_quiet_polls_do_not_read_the_state_file_again(self, tmp_path):
        manager = _manager(tmp_path, _RULE, _fine_until(_RULE, 22))
        with patch.object(
            manager, "_reload_state_for_recovery", wraps=manager._reload_state_for_recovery
        ) as reload_state:
            notify_recovery_cap(manager, _RULE, _at(23, hour=1))
            first = reload_state.call_count
            notify_recovery_cap(manager, _RULE, _at(23, hour=2))
            notify_recovery_cap(manager, _RULE, _at(23, hour=3))

        assert first >= 1
        assert reload_state.call_count == first

    def test_a_new_day_is_looked_at(self, tmp_path):
        # 金曜のうちに抜けが無くても、土曜のポーリングは金曜の抜けを見つける
        manager = _manager(tmp_path, _RULE, _fine_until(_RULE, 22))
        assert _sent_texts(manager, _RULE, _at(23, hour=1)) == []
        assert _FRI in _sent_texts(manager, _RULE, _at(24, hour=1))[0]


@pytest.mark.parametrize("job_id", _JOBS)
class TestRecoveryJudgementIsUnchanged:
    def test_mark_is_not_counted(self, tmp_path, job_id):
        # backfill を付けた形でも、印が前の日の判定を変えないことを見る
        manager = _manager(tmp_path, job_id, _fine_until(job_id, 21), backfill=True)
        now = _at(23)
        missed_before = manager.get_missed_past_periods(job_id, lookback_days=2, now=now)
        recover_before = manager.should_recover(job_id, now=now)

        with patch(_SEND, return_value=True) as send:
            notify_recovery_cap(manager, job_id, now)

        send.assert_called_once()
        assert manager.get_missed_past_periods(job_id, lookback_days=2, now=now) == missed_before
        assert _THU in missed_before
        assert manager.should_recover(job_id, now=now) is recover_before
        assert manager.get_execution_count(job_id, _THU) == 0
        assert manager.has_success(job_id, _THU) is False
        marks = [
            event
            for event in manager._state["events"]
            if event.get("status") == CAP_NOTICE_STATUS and event.get("period_key") == _THU
        ]
        assert len(marks) == 1
        manager.schedule_config[job_id]["func"].assert_not_called()

    def test_no_backfill_stays(self, tmp_path, job_id):
        manager = _manager(tmp_path, job_id, _fine_until(job_id, 21))
        _sent_texts(manager, job_id, _at(23))
        assert manager.get_missed_past_periods(job_id, lookback_days=2, now=_at(23)) == []


class TestOtherJobs:
    @pytest.mark.parametrize(
        "job_id", ["daily_settle_orders", "daily_horizon_exit", "weekly_optimization"]
    )
    def test_past_days_of_other_jobs_are_not_told(self, tmp_path, job_id):
        manager = _manager(tmp_path, job_id, _HISTORY)
        with (
            patch(_SEND, return_value=True) as send,
            patch(
                "src.orchestration.scheduler_cap_notice._load_backlog", return_value=(0, 1, None)
            ),
        ):
            notify_recovery_cap(manager, job_id, _at(23))

        send.assert_not_called()
