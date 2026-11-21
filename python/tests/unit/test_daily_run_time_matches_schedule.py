"""特徴量の待ちの分け方が使う日次の道の時刻が、予定と同じであることを確かめる（#2088）。

src から run_scheduler.py を import できないため、時刻は features_wait.py の DAILY_RUN_TIME と
run_scheduler.py の SCHEDULE_CONFIG["daily_pipeline"] の2か所にある。片方だけを変えると、
日次の道が止まった夜を待ちと取り違えるか、週末の偽の警告が戻るため、ここで揃っていることを見る。
"""

from __future__ import annotations

from datetime import time

import pytest
import run_scheduler

from src.orchestration.jobs import features_wait

_JOB = "daily_pipeline"


def assert_daily_run_time_matches_schedule() -> None:
    """DAILY_RUN_TIME と daily_pipeline の hour・minute が同じでなければ落とす。"""
    config = run_scheduler.SCHEDULE_CONFIG[_JOB]
    scheduled = time(int(config["hour"]), int(config["minute"]))
    assert features_wait.DAILY_RUN_TIME == scheduled, (
        f"features_wait.DAILY_RUN_TIME={features_wait.DAILY_RUN_TIME} と "
        f"run_scheduler の {_JOB} の {scheduled} がずれている。両方を直すこと"
    )


def test_daily_run_time_matches_daily_pipeline_schedule():
    assert_daily_run_time_matches_schedule()


def test_daily_pipeline_schedule_is_weekday_0730():
    """今の予定（mon-fri 07:30）を変えていない。"""
    config = run_scheduler.SCHEDULE_CONFIG[_JOB]
    assert config["day_of_week"] == "mon-fri"
    assert (config["hour"], config["minute"]) == (7, 30)
    assert features_wait.DAILY_RUN_TIME == time(7, 30)


def test_check_fails_when_only_features_wait_time_changes(monkeypatch):
    monkeypatch.setattr(features_wait, "DAILY_RUN_TIME", time(8, 0))
    with pytest.raises(AssertionError, match="ずれている"):
        assert_daily_run_time_matches_schedule()


@pytest.mark.parametrize("key,value", [("hour", 8), ("minute", 45)])
def test_check_fails_when_only_schedule_time_changes(monkeypatch, key, value):
    config = {**run_scheduler.SCHEDULE_CONFIG[_JOB], key: value}
    monkeypatch.setitem(run_scheduler.SCHEDULE_CONFIG, _JOB, config)
    with pytest.raises(AssertionError, match="ずれている"):
        assert_daily_run_time_matches_schedule()
