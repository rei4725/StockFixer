"""消えた回の警告の「補完の対象」を、数えたあとの回数と上限で決める（#1777）

本番の nightly_strategy_factory（上限1・rerun_if_interrupted なし）の 2026-11-05 は、消えた回1つで
上限に届き補完されなかった。weekly_walk_forward_report（上限1・数えない枠1）の 2026-W45 は
消えた回が2つあり、数えない枠を超えた1つで上限に届いた。どちらも警告が補完すると読めてはならない。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import run_scheduler

from src.orchestration.scheduler_interrupted import ExecutionCount, interrupted_recovery_label
from src.orchestration.scheduler_queue import SchedulerQueueManager

_JST = ZoneInfo("Asia/Tokyo")
_FACTORY = "nightly_strategy_factory"
_WF = "weekly_walk_forward_report"


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


def _manager(tmp_path: Path, job_id: str, events: list[dict]) -> SchedulerQueueManager:
    config = dict(run_scheduler.SCHEDULE_CONFIG[job_id])
    config["func"] = MagicMock()
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"events": events}), encoding="utf-8")
    return SchedulerQueueManager({job_id: config}, state_file_path=str(state_path))


def _warnings(caplog) -> list[str]:
    return [r.message for r in caplog.records if "消えた回を見つけました" in r.message]


def test_cap_one_job_without_rerun_says_not_recovered(tmp_path, caplog):
    period_key = "2026-11-05"
    events = [_lost_run(_FACTORY, period_key, 0, "2026-11-04T20:00:00+00:00")]
    manager = _manager(tmp_path, _FACTORY, events)
    now = datetime(2026, 11, 5, 15, 18, tzinfo=_JST)

    with caplog.at_level(logging.WARNING, logger="scheduler"):
        recovered = manager.should_recover(_FACTORY, now=now)

    # 判定は今までどおり。消えた回1つで上限1に届き、補完しない
    assert recovered is False
    lines = _warnings(caplog)
    assert len(lines) == 1
    assert "に数えて補完" not in lines[0]
    assert "上限の1回に届いたので補完しない" in lines[0]


def test_rerun_job_beyond_forgiven_says_not_recovered(tmp_path, caplog):
    period_key = "2026-W45"
    events = [
        _lost_run(_WF, period_key, 0, "2026-11-04T16:00:00+00:00"),
        _lost_run(_WF, period_key, 1, "2026-11-04T17:00:00+00:00"),
    ]
    manager = _manager(tmp_path, _WF, events)
    now = datetime(2026, 11, 5, 15, 18, tzinfo=_JST)

    with caplog.at_level(logging.WARNING, logger="scheduler"):
        recovered = manager.should_recover(_WF, now=now)

    # 数えないのは1つだけ。残りの1つで上限1に届き、補完しない
    assert recovered is False
    lines = _warnings(caplog)
    assert len(lines) == 2
    for line in lines:
        assert "あり（" not in line
        assert "上限の1回に届いたので補完しない" in line
        assert "数えない枠の1回を超えた" in line


def test_rerun_job_within_forgiven_still_says_recovered(tmp_path, caplog):
    period_key = "2026-W45"
    events = [_lost_run(_WF, period_key, 0, "2026-11-04T16:00:00+00:00")]
    manager = _manager(tmp_path, _WF, events)
    now = datetime(2026, 11, 5, 15, 18, tzinfo=_JST)

    with caplog.at_level(logging.WARNING, logger="scheduler"):
        recovered = manager.should_recover(_WF, now=now)

    assert recovered is True
    lines = _warnings(caplog)
    assert len(lines) == 1
    assert "補完の対象=あり（1期間に1回まで）" in lines[0]


def test_label_keeps_recovery_text_below_cap():
    label = interrupted_recovery_label(
        False, 1, 2, ExecutionCount(count=1, interrupted_run_ids=["a"])
    )
    assert label == "上限の2回に数えて補完（#1769）"
