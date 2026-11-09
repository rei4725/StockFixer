"""週次学習の途中でプロセスが消えた回の、同じ週のうちのやり直しと知らせ（#1769）

本番の W46: 2026-11-10 01:00 JST に始まった回が 01:09 に消え、次の起動が
「補完の対象=なし」と記録した。消えた回は上限の2回の1回に数えるだけで、
失敗の回の4時間後（05:00）に同じ火曜のうちに1度やり直す。やり直しも成功しなければ知らせが1通出る。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import run_scheduler

from src.orchestration.scheduler_queue import SchedulerQueueManager

_JST = ZoneInfo("Asia/Tokyo")
_SEND = "src.reporting.discord.discord_utils.send_webhook_text"
_TRAINING = "weekly_model_training"
_WEEK = "2026-W46"
_LOST_START = "2026-11-09T16:00:00+00:00"
_LOST_AT = datetime(2026, 11, 10, 1, 0, tzinfo=_JST)


def _lost_run(index: int, started_at: str = _LOST_START) -> dict:
    return {
        "job_id": _TRAINING,
        "reason": "scheduled",
        "status": "started",
        "period_key": _WEEK,
        "run_id": f"run-{index}",
        "boot_id": f"old-boot-{index}",
        "started_at": started_at,
    }


def _manager(tmp_path: Path, events: list[dict]) -> SchedulerQueueManager:
    config = dict(run_scheduler.SCHEDULE_CONFIG[_TRAINING])
    config["func"] = MagicMock()
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"events": events}), encoding="utf-8")
    return SchedulerQueueManager({_TRAINING: config}, state_file_path=str(state_path))


def test_lost_run_is_redone_on_the_same_tuesday(tmp_path):
    manager = _manager(tmp_path, [_lost_run(0)])

    # 起き直しの 01:09 からは、上限の残りがあっても失敗の回の4時間後まで待つ
    assert manager.should_recover(_TRAINING, now=_LOST_AT + timedelta(minutes=10)) is False
    assert manager.should_recover(_TRAINING, now=_LOST_AT + timedelta(minutes=239)) is False
    # 05:00 に同じ火曜のうちに1度やり直す。07:30 の日次の道の前である
    assert manager.should_recover(_TRAINING, now=_LOST_AT + timedelta(minutes=240)) is True


def test_lost_run_log_says_it_is_redone_within_the_limit(tmp_path, caplog):
    manager = _manager(tmp_path, [_lost_run(0)])

    with caplog.at_level(logging.WARNING, logger="scheduler"):
        manager.should_recover(_TRAINING, now=_LOST_AT + timedelta(minutes=20))

    lines = [r.message for r in caplog.records if "消えた回を見つけました" in r.message]
    assert len(lines) == 1
    # 「なし」だけでは、やり直さないと読める。上限の残りで補完することを書く
    assert "補完の対象=なし" not in lines[0]
    assert "上限の2回に数えて補完" in lines[0]


def test_redo_lost_again_notifies_once(tmp_path):
    redo_start = "2026-11-09T20:00:00+00:00"
    manager = _manager(tmp_path, [_lost_run(0), _lost_run(1, redo_start)])
    now = _LOST_AT + timedelta(hours=5)
    assert manager.should_recover(_TRAINING, now=now) is False

    with patch(_SEND, return_value=True) as send:
        run_scheduler._recover_job(manager, _TRAINING, now)
        run_scheduler._recover_job(manager, _TRAINING, now + timedelta(minutes=5))

    send.assert_called_once()
    text = send.call_args.args[0]
    assert _WEEK in text
    assert "週次モデル学習" in text
