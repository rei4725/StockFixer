"""週次学習と週の Walk-Forward レポートが回数の上限で補完をやめた週の知らせ（#1666）"""

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
_TASKS = "src.watchlist.batch_runner.load_target_symbols"

_TRAINING = "weekly_model_training"
_WALK_FORWARD = "weekly_walk_forward_report"

# 本番の W43 の学習（火曜）と W45 のレポート（木曜）の消え方に、回の時刻を合わせる
_TRAINING_NOW = datetime(2026, 10, 20, 2, 0, tzinfo=_JST)
_WALK_FORWARD_NOW = datetime(2026, 11, 5, 3, 0, tzinfo=_JST)
_WEEK_STARTED_AT = {
    "2026-W36": "2026-09-01T16:00:00+00:00",
    "2026-W43": "2026-10-19T16:00:00+00:00",
    "2026-W45": "2026-11-04T16:00:00+00:00",
}
_WEEKS = [
    pytest.param(_TRAINING, "2026-W43", _TRAINING_NOW, id="training"),
    pytest.param(_WALK_FORWARD, "2026-W45", _WALK_FORWARD_NOW, id="walk_forward"),
]


def _config(job_id: str) -> dict:
    """本番の欄（上限を含む）のまま、関数だけ差し替える。"""
    config = dict(run_scheduler.SCHEDULE_CONFIG[job_id])
    config["func"] = MagicMock()
    return {job_id: config}


def _interrupted(job_id: str, period_key: str, count: int) -> list[dict]:
    # 記録はその週から始まる。前の週は状態ファイルの外で、補完の日が尽きた週に読まない（#1683）
    return [
        {
            "job_id": job_id,
            "reason": "recovery",
            "status": "started",
            "period_key": period_key,
            "run_id": f"run-{index}",
            "boot_id": f"old-boot-{index}",
            "started_at": _WEEK_STARTED_AT[period_key],
        }
        for index in range(count)
    ]


def _success(job_id: str, period_key: str) -> dict:
    return {
        "job_id": job_id,
        "reason": "recovery",
        "status": "success",
        "period_key": period_key,
        "run_id": "run-ok",
        "started_at": "2026-10-19T16:00:00+00:00",
        "finished_at": "2026-10-19T17:00:00+00:00",
        "duration_seconds": 3600.0,
        "error": None,
    }


def _manager(
    tmp_path: Path, job_id: str, events: list[dict], *, write: bool = True
) -> SchedulerQueueManager:
    state_path = tmp_path / "state.json"
    if write:
        state_path.write_text(json.dumps({"events": events}), encoding="utf-8")
    return SchedulerQueueManager(_config(job_id), state_file_path=str(state_path))


class TestCapReached:
    @pytest.mark.parametrize(("job_id", "period_key", "now"), _WEEKS)
    def test_two_lost_runs_notify_once(self, tmp_path, job_id, period_key, now):
        manager = _manager(tmp_path, job_id, _interrupted(job_id, period_key, 2))
        assert manager.should_recover(job_id, now=now) is False

        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, job_id, now)
            run_scheduler._recover_job(manager, job_id, now + timedelta(minutes=5))
            # 再起動の後の新しい起動は、状態ファイルだけを引き継ぐ
            restarted = _manager(tmp_path, job_id, [], write=False)
            run_scheduler._recover_job(restarted, job_id, now + timedelta(hours=1))

        send.assert_called_once()
        text = send.call_args.args[0]
        assert period_key in text
        assert "開始 2 回" in text

    def test_training_message_says_what_stays_old(self, tmp_path):
        manager = _manager(tmp_path, _TRAINING, _interrupted(_TRAINING, "2026-W43", 2))
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, _TRAINING, _TRAINING_NOW)

        text = send.call_args.args[0]
        assert "週次モデル学習" in text
        for word in ("影の比べ", "昇格", "challenger の学習", "精度のチェック"):
            assert word in text

    def test_walk_forward_message_says_what_stays_old(self, tmp_path):
        manager = _manager(tmp_path, _WALK_FORWARD, _interrupted(_WALK_FORWARD, "2026-W45", 2))
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, _WALK_FORWARD, _WALK_FORWARD_NOW)

        text = send.call_args.args[0]
        assert "Walk-Forward" in text
        assert "この週の分が無い" in text

    @pytest.mark.parametrize(("job_id", "period_key", "now"), _WEEKS)
    def test_message_has_no_optimization_backlog(self, tmp_path, job_id, period_key, now):
        manager = _manager(tmp_path, job_id, _interrupted(job_id, period_key, 2))
        with (
            patch(_SEND, return_value=True) as send,
            patch(_TASKS, side_effect=AssertionError("残りの銘柄を数えない")) as tasks,
        ):
            run_scheduler._recover_job(manager, job_id, now)

        tasks.assert_not_called()
        text = send.call_args.args[0]
        assert "銘柄" not in text
        assert "SL/TP" not in text
        assert "週の最適化" not in text

    @pytest.mark.parametrize(("job_id", "period_key", "now"), _WEEKS)
    def test_lost_run_seen_the_next_day_still_notifies(self, tmp_path, job_id, period_key, now):
        # 定時の曜日をまたいで消え、起き直しが翌日の回も、その週のうちに知らせる
        manager = _manager(tmp_path, job_id, _interrupted(job_id, period_key, 2))
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, job_id, now + timedelta(days=1))

        send.assert_called_once()

    @pytest.mark.parametrize(("job_id", "period_key", "now"), _WEEKS)
    def test_notice_record_does_not_change_the_recovery_judgement(
        self, tmp_path, job_id, period_key, now
    ):
        manager = _manager(tmp_path, job_id, _interrupted(job_id, period_key, 2))
        count_before = manager.get_execution_count(job_id, period_key)
        with patch(_SEND, return_value=True):
            run_scheduler._recover_job(manager, job_id, now)

        assert manager.get_execution_count(job_id, period_key) == count_before
        assert manager.should_recover(job_id, now=now) is False
        assert manager.has_success(job_id, period_key) is False


class TestNoNotice:
    @pytest.mark.parametrize(("job_id", "period_key", "now"), _WEEKS)
    def test_week_with_success(self, tmp_path, job_id, period_key, now):
        events = _interrupted(job_id, period_key, 2) + [_success(job_id, period_key)]
        manager = _manager(tmp_path, job_id, events)
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, job_id, now)

        send.assert_not_called()

    @pytest.mark.parametrize(("job_id", "period_key", "now"), _WEEKS)
    def test_week_below_the_cap(self, tmp_path, job_id, period_key, now):
        manager = _manager(tmp_path, job_id, _interrupted(job_id, period_key, 1))
        with (
            patch.object(manager, "should_recover", return_value=False),
            patch(_SEND, return_value=True) as send,
        ):
            run_scheduler._recover_job(manager, job_id, now)

        send.assert_not_called()

    @pytest.mark.parametrize(("job_id", "period_key", "now"), _WEEKS)
    def test_before_the_scheduled_time(self, tmp_path, job_id, period_key, now):
        before = now.replace(hour=0, minute=30)
        manager = _manager(tmp_path, job_id, _interrupted(job_id, period_key, 2))
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, job_id, before)

        send.assert_not_called()

    @pytest.mark.parametrize(("job_id", "period_key", "now"), _WEEKS)
    def test_day_before_the_scheduled_day(self, tmp_path, job_id, period_key, now):
        manager = _manager(tmp_path, job_id, _interrupted(job_id, period_key, 2))
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, job_id, now - timedelta(days=1))

        send.assert_not_called()


def test_optimization_message_is_unchanged(tmp_path):
    job_id = "weekly_optimization"
    now = datetime(2026, 9, 3, 10, 0, tzinfo=_JST)
    manager = _manager(tmp_path, job_id, _interrupted(job_id, "2026-W36", 42))
    with (
        patch("src.orchestration.scheduler_cap_notice._load_backlog", return_value=(2, 3, None)),
        patch(_SEND, return_value=True) as send,
    ):
        run_scheduler._recover_job(manager, job_id, now)

    assert send.call_args.args[0] == (
        "⚠️ 週の最適化の補完が回数の上限で止まった（2026-W36）\n"
        "開始 42 回・失敗 0 回で、成功の回が無い\n"
        "残り 2 / 3 銘柄（あと約 20 時間の見込み）\n"
        "残りの銘柄は前の週の閾値と SL/TP のまま発注に使われる。"
        "次の週の定時の回が古い順に追いつく"
    )
