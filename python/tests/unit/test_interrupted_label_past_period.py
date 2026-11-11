"""消えた回の警告の「補完の対象」を、その期間がもう補完されないときは補完しないと書く（#1783）

本番の 2026-11-07 02:35 JST（土曜）は、W44 の weekly_optimization の消えた回2件（回数0・上限2）に
「補完の対象=あり」と書き、同じポーリングで W44 の補完をやめたと知らせた。W44 の補完の最後の日は
11-01（日）である。補完するかの判定（should_recover・get_missed_past_periods）は変えない。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import run_scheduler

from src.orchestration.scheduler_queue import SchedulerQueueManager

_JST = ZoneInfo("Asia/Tokyo")
_SEND = "src.reporting.discord.discord_utils.send_webhook_text"
_OPT = "weekly_optimization"
_TRAINING = "weekly_model_training"
_BACKUP = "daily_backup"
_ENDED = "補完の日を過ぎたので補完しない"


def _lost_run(job_id: str, period_key: str, index: int, started_at: str) -> dict:
    return {
        "job_id": job_id,
        "reason": "scheduled",
        "status": "started",
        "period_key": period_key,
        "run_id": f"{job_id}-run-{index}",
        "boot_id": f"old-boot-{index}",
        "started_at": started_at,
    }


def _success(job_id: str, period_key: str, started_at: str) -> dict:
    return {
        "job_id": job_id,
        "reason": "scheduled",
        "status": "success",
        "period_key": period_key,
        "run_id": f"{job_id}-ok-{period_key}",
        "started_at": started_at,
        "finished_at": started_at,
    }


def _manager(tmp_path: Path, job_id: str, events: list[dict], now: datetime):
    config = dict(run_scheduler.SCHEDULE_CONFIG[job_id])
    config["func"] = MagicMock()
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"events": events}), encoding="utf-8")
    manager = SchedulerQueueManager({job_id: config}, state_file_path=str(state_path))
    manager.now = lambda: now  # type: ignore[method-assign]
    return manager


def _warnings(caplog) -> list[str]:
    return [r.message for r in caplog.records if "消えた回を見つけました" in r.message]


def _assert_says_not_recovered(line: str) -> None:
    assert "補完の対象=あり" not in line
    assert "に数えて補完" not in line
    assert f"補完の対象=なし（{_ENDED}）" in line


def test_previous_week_optimization_counted_on_saturday_says_not_recovered(tmp_path, caplog):
    events = [
        _lost_run(_OPT, "2026-W44", 0, "2026-10-27T16:00:00+00:00"),
        _lost_run(_OPT, "2026-W44", 1, "2026-10-28T04:00:00+00:00"),
        _success(_OPT, "2026-W45", "2026-11-03T16:00:00+00:00"),
    ]
    saturday = datetime(2026, 11, 7, 2, 35, tzinfo=_JST)
    manager = _manager(tmp_path, _OPT, events, saturday)

    with caplog.at_level(logging.WARNING, logger="scheduler"):
        with patch(_SEND, return_value=True) as send:
            run_scheduler._recover_job(manager, _OPT, saturday)

    # 判定は今までどおり。W45 は済み、W44 は補完の日が尽きたと知らせる
    send.assert_called_once()
    assert "2026-W44" in send.call_args.args[0]
    lines = _warnings(caplog)
    assert len(lines) == 2
    for line in lines:
        assert "period_key=2026-W44" in line
        _assert_says_not_recovered(line)
    # 回数は消えた回を数えないまま
    assert manager.get_execution_count(_OPT, "2026-W44") == 0


def test_training_counted_after_tuesday_says_not_recovered(tmp_path, caplog):
    events = [_lost_run(_TRAINING, "2026-W46", 0, "2026-11-09T16:00:03+00:00")]
    wednesday = datetime(2026, 11, 11, 1, 0, tzinfo=_JST)
    manager = _manager(tmp_path, _TRAINING, events, wednesday)

    with caplog.at_level(logging.WARNING, logger="scheduler"):
        with patch(_SEND, return_value=True):
            assert manager.should_recover(_TRAINING, now=wednesday) is False
            run_scheduler._recover_job(manager, _TRAINING, wednesday)

    lines = _warnings(caplog)
    assert len(lines) == 1
    _assert_says_not_recovered(lines[0])


def test_training_counted_on_tuesday_still_says_recovered(tmp_path, caplog):
    events = [_lost_run(_TRAINING, "2026-W46", 0, "2026-11-09T16:00:03+00:00")]
    tuesday = datetime(2026, 11, 10, 1, 20, tzinfo=_JST)
    manager = _manager(tmp_path, _TRAINING, events, tuesday)

    with caplog.at_level(logging.WARNING, logger="scheduler"):
        manager.should_recover(_TRAINING, now=tuesday)

    lines = _warnings(caplog)
    assert len(lines) == 1
    assert "補完の対象=上限の2回に数えて補完（#1769）" in lines[0]


def test_yesterday_backfill_still_says_recovered(tmp_path, caplog):
    events = [_lost_run(_BACKUP, "2026-10-19", 0, "2026-10-18T18:30:00+00:00")]
    now = datetime(2026, 10, 20, 2, 0, tzinfo=_JST)
    manager = _manager(tmp_path, _BACKUP, events, now)

    with caplog.at_level(logging.WARNING, logger="scheduler"):
        missed = manager.get_missed_past_periods(_BACKUP, now=now)

    assert missed == ["2026-10-19"]
    lines = _warnings(caplog)
    assert len(lines) == 1
    assert _ENDED not in lines[0]


def test_day_beyond_backfill_says_not_recovered(tmp_path, caplog):
    events = [_lost_run(_BACKUP, "2026-10-16", 0, "2026-10-15T18:30:00+00:00")]
    now = datetime(2026, 10, 20, 2, 0, tzinfo=_JST)
    manager = _manager(tmp_path, _BACKUP, events, now)

    with caplog.at_level(logging.WARNING, logger="scheduler"):
        count = manager.get_execution_count(_BACKUP, "2026-10-16")

    assert "2026-10-16" not in manager.get_missed_past_periods(_BACKUP, now=now)
    assert count == manager.get_execution_count(_BACKUP, "2026-10-16")
    lines = _warnings(caplog)
    assert len(lines) == 1
    _assert_says_not_recovered(lines[0])
