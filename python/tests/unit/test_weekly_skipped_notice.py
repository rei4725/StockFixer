"""週次ジョブの補完の日が尽きた週に、成功が無ければ回数が上限の下でも知らせる（#1683）"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
import run_scheduler

from src.orchestration.scheduler_cap_notice import CAP_NOTICE_STATUS
from src.orchestration.scheduler_queue import SchedulerQueueManager

_JST = ZoneInfo("Asia/Tokyo")
_SEND = "src.reporting.discord.discord_utils.send_webhook_text"
_BACKLOG = "src.orchestration.scheduler_cap_notice._load_backlog"

_TRAINING = "weekly_model_training"
_RULE = "weekly_rule_evaluation"
_REGIME = "regime_leverage_weekly"
_OPTIMIZATION = "weekly_optimization"
_WALK_FORWARD = "weekly_walk_forward_report"

# 2026-11-02 が W45 の月曜。W44 の土曜が 10-31、日曜が 11-01
_TRAINING_POLL = datetime(2026, 11, 4, 0, 5, tzinfo=_JST)  # W45 の水曜
_MONDAY_POLL = datetime(2026, 11, 2, 0, 5, tzinfo=_JST)  # W45 の月曜。W44 を見る

_ENDED = [
    pytest.param(_TRAINING, "2026-W45", _TRAINING_POLL, id="training"),
    pytest.param(_RULE, "2026-W44", _MONDAY_POLL, id="rule"),
    pytest.param(_REGIME, "2026-W44", _MONDAY_POLL, id="regime"),
]

_DAY_BEFORE_WEEK = {"2026-W44": "2026-10-25", "2026-W45": "2026-11-01"}

# 補完の最後の日の終わり。補完はここまで動ける
_LAST_DAY = [
    pytest.param(_TRAINING, datetime(2026, 11, 3, 23, 55, tzinfo=_JST), id="training"),
    pytest.param(_RULE, datetime(2026, 11, 1, 23, 55, tzinfo=_JST), id="rule"),
    pytest.param(_REGIME, datetime(2026, 11, 1, 23, 55, tzinfo=_JST), id="regime"),
    pytest.param(_OPTIMIZATION, datetime(2026, 11, 8, 23, 55, tzinfo=_JST), id="optimization"),
]


def _config(job_id: str, **overrides: object) -> dict:
    """本番の欄のまま、関数だけ差し替える。"""
    config = dict(run_scheduler.SCHEDULE_CONFIG[job_id])
    config["func"] = MagicMock()
    config.update(overrides)
    return {job_id: config}


def _old_record(day: str = "2026-10-01") -> dict:
    """見る週より前の記録。状態ファイルがその週を保持していることを表す。"""
    return {
        "job_id": "daily_pipeline",
        "reason": "scheduled",
        "status": "success",
        "period_key": day,
        "run_id": f"old-{day}",
        "started_at": f"{day}T00:00:00+00:00",
        "finished_at": f"{day}T00:10:00+00:00",
    }


def _finished(job_id: str, period_key: str, status: str, run_id: str = "run-a") -> dict:
    return {
        "job_id": job_id,
        "reason": "scheduled",
        "status": status,
        "period_key": period_key,
        "run_id": run_id,
        "started_at": "2026-10-31T00:00:00+00:00",
        "finished_at": "2026-10-31T00:10:00+00:00",
        "duration_seconds": 600.0,
        "error": None,
    }


def _held(period_key: str) -> dict:
    """その週から読める、最も古い記録。その前の週は状態ファイルの外になる。"""
    return _old_record(_DAY_BEFORE_WEEK[period_key])


def _failed_run(job_id: str, period_key: str) -> list[dict]:
    started = {**_finished(job_id, period_key, "started"), "boot_id": "old-boot"}
    return [started, _finished(job_id, period_key, "error")]


def _manager(
    tmp_path: Path, job_id: str, events: list[dict], *, write: bool = True, **overrides: object
) -> SchedulerQueueManager:
    state_path = tmp_path / "state.json"
    if write:
        state_path.write_text(json.dumps({"events": events}), encoding="utf-8")
    return SchedulerQueueManager(_config(job_id, **overrides), state_file_path=str(state_path))


def _marks(manager: SchedulerQueueManager, job_id: str) -> list[str]:
    manager._reload_state_for_recovery()
    return [
        event["period_key"]
        for event in manager._state["events"]
        if event["job_id"] == job_id and event["status"] == CAP_NOTICE_STATUS
    ]


class TestEndedWeekWithoutSuccess:
    @pytest.mark.parametrize(("job_id", "period_key", "now"), _ENDED)
    def test_no_run_at_all_notifies_once(self, tmp_path, job_id, period_key, now):
        manager = _manager(tmp_path, job_id, [_held(period_key)])
        assert manager.should_recover(job_id, now=now) is False

        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, job_id, now)
            run_scheduler._recover_job(manager, job_id, now + timedelta(minutes=5))

        send.assert_called_once()
        text = send.call_args.args[0]
        assert period_key in text
        assert "補完の日が尽きた" in text
        assert "回数の上限" not in text
        assert "動かなかった（開始 0 回・失敗 0 回）" in text

    @pytest.mark.parametrize(("job_id", "period_key", "now"), _ENDED)
    def test_one_failed_run_below_the_cap_notifies(self, tmp_path, job_id, period_key, now):
        manager = _manager(tmp_path, job_id, [_held(period_key), *_failed_run(job_id, period_key)])
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, job_id, now)

        send.assert_called_once()
        text = send.call_args.args[0]
        assert "補完の日が尽きた" in text
        assert "失敗した（開始 1 回・失敗 1 回）" in text

    def test_first_failure_late_on_tuesday_with_a_gap_is_notified(self, tmp_path):
        # 間（PR 1687 の 240 分）が水曜へ越え、補完が走らず上限の下のまま定時の曜日を過ぎた週
        failed = {
            **_finished(_TRAINING, "2026-W45", "error"),
            "started_at": "2026-11-03T13:00:00+00:00",
            "finished_at": "2026-11-03T13:30:00+00:00",  # JST で火曜 22:30
        }
        manager = _manager(
            tmp_path, _TRAINING, [_held("2026-W45"), failed], recovery_min_gap_minutes=240
        )
        tuesday_night = datetime(2026, 11, 3, 23, 55, tzinfo=_JST)
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, _TRAINING, tuesday_night)
            send.assert_not_called()
            run_scheduler._recover_job(manager, _TRAINING, _TRAINING_POLL)

        send.assert_called_once()
        assert "失敗 1 回" in send.call_args.args[0]

    def test_week_seen_two_days_late_is_the_same_week(self, tmp_path):
        # W38 と同じ金曜の起き直し
        friday = datetime(2026, 11, 6, 22, 10, tzinfo=_JST)
        manager = _manager(tmp_path, _TRAINING, [_held("2026-W45")])
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, _TRAINING, friday)

        send.assert_called_once()
        assert "2026-W45" in send.call_args.args[0]

    def test_optimization_week_ended_says_the_backlog(self, tmp_path):
        manager = _manager(tmp_path, _OPTIMIZATION, [_held("2026-W45")])
        monday = datetime(2026, 11, 9, 0, 5, tzinfo=_JST)
        with patch(_BACKLOG, return_value=(2, 3, None)), patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, _OPTIMIZATION, monday)

        send.assert_called_once()
        text = send.call_args.args[0]
        assert "週の最適化の回が成功しないまま補完の日が尽きた（2026-W45）" in text
        assert "残り 2 / 3 銘柄" in text

    def test_previous_week_at_the_cap_keeps_the_cap_headline(self, tmp_path):
        # 上限に届いたまま週が変わり、週のうちに知らせられなかった
        events = [_old_record()] + [
            _finished(_REGIME, "2026-W44", "error", f"run-{index}") for index in range(12)
        ]
        manager = _manager(tmp_path, _REGIME, events)
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, _REGIME, _MONDAY_POLL)

        send.assert_called_once()
        assert "レジーム" in send.call_args.args[0]
        assert "回数の上限で止まった（2026-W44）" in send.call_args.args[0]

    def test_two_ended_weeks_are_two_notices_oldest_first(self, tmp_path):
        manager = _manager(tmp_path, _TRAINING, [_old_record()])
        friday = datetime(2026, 11, 6, 12, 0, tzinfo=_JST)
        with patch(_SEND, return_value=True) as send:
            for minutes in (0, 5, 10):
                run_scheduler._recover_job(manager, _TRAINING, friday + timedelta(minutes=minutes))

        sent = [call.args[0] for call in send.call_args_list]
        assert len(sent) == 2
        assert "2026-W44" in sent[0]
        assert "2026-W45" in sent[1]

    def test_failed_send_is_retried_by_the_next_poll(self, tmp_path):
        manager = _manager(tmp_path, _RULE, [_old_record()])
        with patch(_SEND, return_value=False):
            run_scheduler._recover_job(manager, _RULE, _MONDAY_POLL)
        assert _marks(manager, _RULE) == []

        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, _RULE, _MONDAY_POLL + timedelta(minutes=5))

        send.assert_called_once()
        assert _marks(manager, _RULE) == ["2026-W44"]

    @pytest.mark.parametrize(
        ("job_id", "words"),
        [
            pytest.param(_RULE, ("rule_best_by_symbol", "16:30", "次の日曜"), id="rule"),
            pytest.param(_REGIME, ("レジーム転換", "決済", "次の土曜"), id="regime"),
        ],
    )
    def test_message_says_what_stays_old(self, tmp_path, job_id, words):
        manager = _manager(tmp_path, job_id, [_old_record()])
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, job_id, _MONDAY_POLL)

        for word in words:
            assert word in send.call_args.args[0]


class TestNoNotice:
    @pytest.mark.parametrize(("job_id", "now"), _LAST_DAY)
    def test_while_recovery_can_still_run(self, tmp_path, job_id, now):
        previous = now - timedelta(days=7)
        week = f"{previous.isocalendar().year}-W{previous.isocalendar().week:02d}"
        events = [_old_record(), _finished(job_id, week, "success")]
        manager = _manager(tmp_path, job_id, events)
        with (
            patch.object(manager, "should_recover", return_value=False),
            patch(_SEND, return_value=True) as send,
        ):
            run_scheduler._recover_job(manager, job_id, now)

        send.assert_not_called()

    @pytest.mark.parametrize(("job_id", "period_key", "now"), _ENDED)
    def test_week_with_success(self, tmp_path, job_id, period_key, now):
        events = [
            _held(period_key),
            _finished(job_id, period_key, "error"),
            _finished(job_id, period_key, "success", "run-b"),
        ]
        manager = _manager(tmp_path, job_id, events)
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, job_id, now)

        send.assert_not_called()

    @pytest.mark.parametrize(
        ("job_id", "now", "previous_week"),
        [
            pytest.param(_TRAINING, datetime(2026, 11, 2, 12, 0, tzinfo=_JST), "2026-W44"),
            pytest.param(_RULE, datetime(2026, 11, 1, 1, 0, tzinfo=_JST), "2026-W43"),
            pytest.param(_REGIME, datetime(2026, 10, 31, 6, 0, tzinfo=_JST), "2026-W43"),
        ],
    )
    def test_before_the_scheduled_time(self, tmp_path, job_id, now, previous_week):
        events = [_old_record(), _finished(job_id, previous_week, "success")]
        manager = _manager(tmp_path, job_id, events)
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, job_id, now)

        send.assert_not_called()

    @pytest.mark.parametrize(("job_id", "period_key", "now"), _ENDED)
    def test_run_in_progress(self, tmp_path, job_id, period_key, now):
        # 月曜へまたいで動いている回が成功するかは、まだ分からない
        manager = _manager(tmp_path, job_id, [_old_record()])
        manager._in_progress_jobs.add(job_id)
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, job_id, now)

        send.assert_not_called()

    def test_weekly_report_is_not_covered(self, tmp_path):
        manager = _manager(tmp_path, "weekly_report", [_old_record()])
        friday = datetime(2026, 11, 6, 12, 0, tzinfo=_JST)
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, "weekly_report", friday)

        send.assert_not_called()


class TestSecondNotice:
    @pytest.mark.parametrize(("job_id", "period_key", "now"), _ENDED)
    def test_restart_does_not_send_again(self, tmp_path, job_id, period_key, now):
        manager = _manager(tmp_path, job_id, [_held(period_key)])
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, job_id, now)
            # 再起動の後の新しい起動は、状態ファイルだけを引き継ぐ
            restarted = _manager(tmp_path, job_id, [], write=False)
            run_scheduler._recover_job(restarted, job_id, now + timedelta(hours=1))

        send.assert_called_once()
        assert _marks(restarted, job_id) == [period_key]

    def test_restart_that_forgives_a_lost_run_does_not_send_again(self, tmp_path):
        # 11-06 22:10 の起き直しの形。上限で知らせた週が、起き直しで消えた回を許されて
        # 上限の下へ戻っても、同じ週の2通目を出さない
        first = _manager(tmp_path, _WALK_FORWARD, [_held("2026-W45")])
        lost = {
            "job_id": _WALK_FORWARD,
            "reason": "scheduled",
            "status": "started",
            "period_key": "2026-W45",
            "run_id": "run-lost",
            "boot_id": first._boot_id,
            "started_at": "2026-11-05T13:00:00+00:00",
        }
        first._append_event(lost)
        friday = datetime(2026, 11, 6, 20, 0, tzinfo=_JST)
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(first, _WALK_FORWARD, friday)
            send.assert_called_once()
            assert "回数の上限で止まった" in send.call_args.args[0]

            restarted = _manager(tmp_path, _WALK_FORWARD, [], write=False)
            assert restarted.get_execution_count(_WALK_FORWARD, "2026-W45") == 0
            run_scheduler._recover_job(
                restarted, _WALK_FORWARD, friday + timedelta(hours=2, minutes=10)
            )

        send.assert_called_once()

    @pytest.mark.parametrize(
        ("job_id", "period_key", "capped_at", "now"),
        [
            pytest.param(
                _TRAINING,
                "2026-W45",
                datetime(2026, 11, 3, 12, 0, tzinfo=_JST),
                _TRAINING_POLL,
                id="training",
            ),
            pytest.param(
                _RULE,
                "2026-W44",
                datetime(2026, 11, 1, 12, 0, tzinfo=_JST),
                _MONDAY_POLL,
                id="rule",
            ),
            pytest.param(
                _REGIME,
                "2026-W44",
                datetime(2026, 11, 1, 12, 0, tzinfo=_JST),
                _MONDAY_POLL,
                id="regime",
            ),
        ],
    )
    def test_week_already_noticed_at_the_cap(self, tmp_path, job_id, period_key, capped_at, now):
        events = [_held(period_key)] + [
            _finished(job_id, period_key, "error", f"run-{index}") for index in range(12)
        ]
        manager = _manager(tmp_path, job_id, events)
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, job_id, capped_at)
            send.assert_called_once()
            assert "回数の上限で止まった" in send.call_args.args[0]
            run_scheduler._recover_job(manager, job_id, now)

        send.assert_called_once()


class TestOutsideTheRecords:
    @pytest.mark.parametrize(("job_id", "period_key", "now"), _ENDED)
    def test_empty_state_is_not_a_week_without_runs(self, tmp_path, job_id, period_key, now):
        manager = _manager(tmp_path, job_id, [])
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, job_id, now)

        send.assert_not_called()

    @pytest.mark.parametrize(
        ("job_id", "now", "oldest"),
        [
            pytest.param(_TRAINING, _TRAINING_POLL, "2026-11-03", id="training"),
            pytest.param(_RULE, _MONDAY_POLL, "2026-10-28", id="rule"),
            pytest.param(_REGIME, _MONDAY_POLL, "2026-10-28", id="regime"),
        ],
    )
    def test_week_cut_off_the_head_of_the_records(self, tmp_path, job_id, now, oldest):
        manager = _manager(tmp_path, job_id, [_old_record(oldest)])
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, job_id, now)

        send.assert_not_called()

    def test_oldest_record_on_the_monday_may_be_cut(self, tmp_path):
        # 最も古い記録の日は頭が消えているかもしれないので、その日の週は読まない
        manager = _manager(tmp_path, _RULE, [_old_record("2026-10-26")])
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, _RULE, _MONDAY_POLL)

        send.assert_not_called()

    def test_week_after_the_oldest_record_is_read(self, tmp_path):
        manager = _manager(tmp_path, _RULE, [_old_record("2026-10-25")])
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, _RULE, _MONDAY_POLL)

        send.assert_called_once()


class TestRecoveryJudgementIsUnchanged:
    @pytest.mark.parametrize(("job_id", "period_key", "now"), _ENDED)
    def test_mark_counts_for_nothing(self, tmp_path, job_id, period_key, now):
        manager = _manager(tmp_path, job_id, [_held(period_key), *_failed_run(job_id, period_key)])
        before = (
            manager.get_execution_count(job_id, period_key),
            manager.should_recover(job_id, now=now),
            manager.get_missed_past_periods(job_id, now=now),
        )
        with patch(_SEND, return_value=True):
            run_scheduler._recover_job(manager, job_id, now)
        assert _marks(manager, job_id) == [period_key]

        assert manager.has_success(job_id, period_key) is False
        assert before == (
            manager.get_execution_count(job_id, period_key),
            manager.should_recover(job_id, now=now),
            manager.get_missed_past_periods(job_id, now=now),
        )

    @pytest.mark.parametrize(
        ("job_id", "period_key", "last_day"),
        [
            pytest.param(_TRAINING, "2026-W45", datetime(2026, 11, 3, 12, 0, tzinfo=_JST)),
            pytest.param(_RULE, "2026-W44", datetime(2026, 11, 1, 12, 0, tzinfo=_JST)),
            pytest.param(_REGIME, "2026-W44", datetime(2026, 11, 1, 12, 0, tzinfo=_JST)),
        ],
    )
    def test_mark_does_not_stop_a_recovery_that_was_due(
        self, tmp_path, job_id, period_key, last_day
    ):
        mark = {
            "job_id": job_id,
            "reason": "recovery_cap",
            "status": CAP_NOTICE_STATUS,
            "period_key": period_key,
            "started_at": "2026-11-01T00:00:00+00:00",
            "finished_at": "2026-11-01T00:00:00+00:00",
        }
        manager = _manager(tmp_path, job_id, [_held(period_key), mark])

        assert manager.should_recover(job_id, now=last_day) is True
