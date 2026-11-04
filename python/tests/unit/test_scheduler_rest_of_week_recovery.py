"""週の定時の曜日より後の、同じ ISO 週の補完（#1534）"""

import json
import os
import sys
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.orchestration.scheduler_queue import SchedulerQueueManager  # noqa: E402

_JOB_ID = "weekly_optimization"
# 2026-11-02 は月曜。2026-W45
_WEEK_KEY = "2026-W45"


def _write_events(state_path, events: list[dict]) -> None:
    state_path.write_text(json.dumps({"events": events}), encoding="utf-8")


def _interrupted(period_key: str, index: int) -> dict:
    """前の起動のプロセスが始めの記録だけを残して消えた回"""
    return {
        "job_id": _JOB_ID,
        "reason": "scheduled",
        "status": "started",
        "period_key": period_key,
        "run_id": f"run-{index}",
        "boot_id": f"boot-{index}",
        "started_at": "2020-01-01T00:00:00+00:00",
    }


def _manager(tmp_path, events: list[dict] | None = None) -> SchedulerQueueManager:
    from run_scheduler import SCHEDULE_CONFIG

    state_path = tmp_path / "state.json"
    _write_events(state_path, events or [])
    return SchedulerQueueManager(SCHEDULE_CONFIG, state_file_path=str(state_path))


@pytest.mark.parametrize(
    "day, hour, minute",
    [
        (5, 0, 30),  # 木曜 0:30。その日の定時の 1:00 より前でも水曜の定時は過ぎている
        (5, 12, 0),  # 木曜
        (6, 12, 0),  # 金曜
        (7, 12, 0),  # 土曜
        (8, 23, 59),  # 日曜の終わり
    ],
)
def test_interrupted_run_is_recovered_on_days_after_scheduled_day(tmp_path, day, hour, minute):
    """水曜に消えた回を、同じ週の木〜日に補完すること"""
    manager = _manager(tmp_path, [_interrupted(_WEEK_KEY, 0)])

    now = datetime(2026, 11, day, hour, minute, tzinfo=manager.tz)
    assert manager.should_recover(_JOB_ID, now=now) is True


def test_week_without_any_run_is_recovered_after_scheduled_day(tmp_path):
    """定時の回が1度も動かなかった週も、木曜に補完すること"""
    manager = _manager(tmp_path)

    now = datetime(2026, 11, 5, 12, 0, tzinfo=manager.tz)
    assert manager.should_recover(_JOB_ID, now=now) is True


def test_week_with_success_is_not_recovered_after_scheduled_day(tmp_path):
    """その週の成功があれば、木曜も補完しないこと"""
    events = [
        _interrupted(_WEEK_KEY, 0),
        {"job_id": _JOB_ID, "status": "success", "period_key": _WEEK_KEY, "run_id": "run-ok"},
    ]
    manager = _manager(tmp_path, events)

    now = datetime(2026, 11, 5, 12, 0, tzinfo=manager.tz)
    assert manager.should_recover(_JOB_ID, now=now) is False


def test_week_at_limit_is_not_recovered_after_scheduled_day(tmp_path):
    """消えた回が上限を超えた週は、木曜も補完しないこと"""
    from run_scheduler import SCHEDULE_CONFIG

    limit = SCHEDULE_CONFIG[_JOB_ID]["max_interrupted_reruns_per_period"]
    events = [_interrupted(_WEEK_KEY, i) for i in range(limit + 1)]
    manager = _manager(tmp_path, events)

    now = datetime(2026, 11, 5, 12, 0, tzinfo=manager.tz)
    assert manager.should_recover(_JOB_ID, now=now) is False
    assert manager.get_execution_count(_JOB_ID, _WEEK_KEY) == 1


@pytest.mark.parametrize("day, hour", [(2, 12), (3, 23)])
def test_days_before_scheduled_day_are_not_recovered(tmp_path, day, hour):
    """月曜・火曜は週の定時より前なので、同じ週の消えた回があっても補完しないこと"""
    manager = _manager(tmp_path, [_interrupted(_WEEK_KEY, 0)])

    now = datetime(2026, 11, day, hour, 0, tzinfo=manager.tz)
    assert manager.should_recover(_JOB_ID, now=now) is False


def test_scheduled_day_before_scheduled_time_is_not_recovered(tmp_path):
    """定時の曜日でも、定時と猶予を過ぎるまでは今までどおり補完しないこと"""
    manager = _manager(tmp_path)

    before_delay = datetime(2026, 11, 4, 1, 29, tzinfo=manager.tz)
    after_delay = datetime(2026, 11, 4, 1, 30, tzinfo=manager.tz)
    assert manager.should_recover(_JOB_ID, now=before_delay) is False
    assert manager.should_recover(_JOB_ID, now=after_delay) is True


def test_next_week_does_not_inherit_previous_week_failures(tmp_path):
    """前の週の回は次の週に持ち越さず、次の週の月曜は補完しないこと"""
    manager = _manager(tmp_path, [_interrupted(_WEEK_KEY, 0)])

    now = datetime(2026, 11, 9, 12, 0, tzinfo=manager.tz)
    assert manager.should_recover(_JOB_ID, now=now) is False


def test_recovery_after_scheduled_day_uses_same_period_key(tmp_path):
    """木曜の補完の回が水曜の回と同じ週に数えられ、成功の後は止まること"""
    from run_scheduler import SCHEDULE_CONFIG

    calls = {"n": 0}

    def _succeed():
        calls["n"] += 1

    config = {_JOB_ID: {**SCHEDULE_CONFIG[_JOB_ID], "func": _succeed, "max_retries": 0}}
    state_path = tmp_path / "state.json"
    _write_events(state_path, [_interrupted(_WEEK_KEY, 0)])
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))

    now = datetime(2026, 11, 7, 12, 0, tzinfo=manager.tz)
    manager.now = lambda: now  # type: ignore[method-assign]
    assert manager.should_recover(_JOB_ID, now=now) is True
    assert manager.run_job(_JOB_ID, reason="recovery") is True

    assert calls["n"] == 1
    assert manager.has_success(_JOB_ID, _WEEK_KEY) is True
    assert manager.should_recover(_JOB_ID, now=now) is False


def test_iso_week_across_new_year_counts_as_one_week(tmp_path):
    """年をまたぐ ISO 週でも、水曜の回の鍵のまま土曜に補完すること"""
    # 2026-12-30 は水曜、2027-01-02 は土曜。どちらも 2026-W53
    manager = _manager(tmp_path, [_interrupted("2026-W53", 0)])
    config = manager.schedule_config[_JOB_ID]

    wednesday = datetime(2026, 12, 30, 2, 0, tzinfo=manager.tz)
    saturday = datetime(2027, 1, 2, 12, 0, tzinfo=manager.tz)
    assert manager._build_period_key(config, wednesday) == "2026-W53"
    assert manager._build_period_key(config, saturday) == "2026-W53"
    assert manager.should_recover(_JOB_ID, now=saturday) is True

    limit = config["max_interrupted_reruns_per_period"]
    events = [_interrupted("2026-W53", i) for i in range(limit + 1)]
    manager = _manager(tmp_path, events)
    assert manager.should_recover(_JOB_ID, now=saturday) is False


@pytest.mark.parametrize("hours_between_boots", [1.9, 0.82])
def test_limit_gives_enough_time_for_all_symbols(hours_between_boots):
    """起動の間隔が W44 の水曜の速さ（22回で18時間）でも、全銘柄の約29時間に届く回数であること"""
    from run_scheduler import SCHEDULE_CONFIG

    runs = 1 + SCHEDULE_CONFIG[_JOB_ID]["max_interrupted_reruns_per_period"]

    assert timedelta(hours=hours_between_boots) * runs > timedelta(hours=29)


def test_weekly_job_without_flag_is_only_recovered_on_scheduled_day(tmp_path):
    """印の無い週次ジョブは、定時の曜日のほかで今までどおり補完しないこと"""
    from run_scheduler import SCHEDULE_CONFIG

    job_id = "weekly_walk_forward_report"  # 木曜 01:00
    assert not SCHEDULE_CONFIG[job_id].get("recover_rest_of_week", False)
    state_path = tmp_path / "state.json"
    _write_events(state_path, [])
    manager = SchedulerQueueManager(SCHEDULE_CONFIG, state_file_path=str(state_path))

    scheduled_day = datetime(2026, 11, 5, 2, 0, tzinfo=manager.tz)
    assert manager.should_recover(job_id, now=scheduled_day) is True
    for day in (2, 3, 4, 6, 7, 8):
        now = datetime(2026, 11, day, 12, 0, tzinfo=manager.tz)
        assert manager.should_recover(job_id, now=now) is False


def test_flag_only_on_listed_weekly_jobs():
    """印を持つのは週の最適化とレジームレバレッジの週次判定だけで、ほかの判定は変わらないこと"""
    from run_scheduler import SCHEDULE_CONFIG

    flagged = {
        job_id for job_id, config in SCHEDULE_CONFIG.items() if config.get("recover_rest_of_week")
    }
    assert flagged == {_JOB_ID, "regime_leverage_weekly"}


def test_flag_is_ignored_on_non_weekly_job(tmp_path):
    """印を週次でないジョブに付けても、定時の曜日のほかで補完しないこと"""
    config = {
        "daily_job": {
            "func": lambda: None,
            "period": "daily",
            "day_of_week": "wed",
            "hour": 1,
            "minute": 0,
            "recover_rest_of_week": True,
        }
    }
    manager = SchedulerQueueManager(config, state_file_path=str(tmp_path / "state.json"))

    now = datetime(2026, 11, 5, 12, 0, tzinfo=manager.tz)
    assert manager.should_recover("daily_job", now=now) is False
