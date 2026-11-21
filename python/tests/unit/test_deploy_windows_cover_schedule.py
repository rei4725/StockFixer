"""deploy_windows.json の見送りの窓が SCHEDULE_CONFIG の守るジョブを覆うことを確かめる (#2090)。

窓の表は SCHEDULE_CONFIG の時刻を人が写した値である。ジョブの時刻を動かして
窓を直し忘れると、自動デプロイの起き直しが発注などを消す。ここでは、守るジョブの
定時のそれぞれについて、定時の20分前から定時まで（両端を含む分）が、その曜日に
効く窓で切れ目なく覆われることを確かめる。

窓の読み方は deploy_windows.json の _note のとおり。時刻は JST の HH:mm で end は
含まない。daily が true の窓は毎日、day がある窓はその曜日だけ、どちらも無い窓は
平日（月〜金）だけ効く。
"""

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import run_scheduler

_PATH = Path(__file__).resolve().parents[3] / "deploy_windows.json"
_LEAD_MINUTES = 20
_DAY_MINUTES = 24 * 60

# 見送りの窓で守るジョブ。今の表で覆われないジョブは、理由のコメントつきで外す。
_PROTECTED_JOBS = (
    "daily_pipeline",
    "daily_auto_order",
    "daily_horizon_exit",
    "daily_settle_orders",
    "daily_settle_orders_retry",
    "daily_rule_signals",
    "nightly_backtest_promotion",
    "weekly_model_training",
    "weekly_walk_forward_report",
)

_CRON_DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_WINDOW_DAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


def _load_windows() -> list[dict[str, Any]]:
    if not _PATH.exists():
        pytest.skip("deploy_windows.json が無い環境")
    windows: list[dict[str, Any]] = json.loads(_PATH.read_text(encoding="utf-8"))["windows"]
    return windows


def _parse_cron_days(spec: str) -> set[int]:
    days: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            days.update(range(_CRON_DAYS.index(lo), _CRON_DAYS.index(hi) + 1))
        else:
            days.add(_CRON_DAYS.index(part))
    return days


def _parse_int_list(value: int | str) -> list[int]:
    if isinstance(value, int):
        return [value]
    return [int(v) for v in value.split(",")]


def _hhmm_to_minutes(text: str) -> int:
    hour, minute = text.split(":")
    return int(hour) * 60 + int(minute)


def _window_applies(window: dict[str, Any], weekday: int) -> bool:
    if window.get("daily") is True:
        return True
    if "day" in window:
        return bool(window["day"] == _WINDOW_DAYS[weekday])
    return weekday < 5


def _is_covered(windows: list[dict[str, Any]], weekday: int, minute_of_day: int) -> bool:
    return any(
        _window_applies(w, weekday)
        and _hhmm_to_minutes(w["start"]) <= minute_of_day < _hhmm_to_minutes(w["end"])
        for w in windows
    )


def _uncovered(windows: list[dict[str, Any]], schedule: dict[str, Any]) -> list[str]:
    """守るジョブの定時の20分前から定時までで、窓に覆われない分を返す。"""
    gaps: list[str] = []
    for name in _PROTECTED_JOBS:
        job = schedule[name]
        assert job["trigger"] == "cron", name
        for weekday in sorted(_parse_cron_days(job["day_of_week"])):
            for hour in _parse_int_list(job["hour"]):
                for minute in _parse_int_list(job["minute"]):
                    fire = hour * 60 + minute
                    for offset in range(-_LEAD_MINUTES, 1):
                        at = fire + offset
                        day = weekday
                        if at < 0:
                            at += _DAY_MINUTES
                            day = (weekday - 1) % 7
                        if not _is_covered(windows, day, at):
                            gaps.append(
                                f"{name} {_CRON_DAYS[weekday]} {hour:02d}:{minute:02d}"
                                f" -> {_CRON_DAYS[day]} {at // 60:02d}:{at % 60:02d}"
                            )
    return gaps


def test_protected_jobs_exist_in_schedule_config() -> None:
    for name in _PROTECTED_JOBS:
        assert name in run_scheduler.SCHEDULE_CONFIG, name


def test_windows_cover_protected_jobs() -> None:
    windows = _load_windows()
    assert _uncovered(windows, run_scheduler.SCHEDULE_CONFIG) == []


def test_moving_job_time_breaks_cover() -> None:
    windows = _load_windows()
    schedule = copy.deepcopy(run_scheduler.SCHEDULE_CONFIG)
    schedule["daily_rule_signals"]["hour"] = 17
    schedule["daily_rule_signals"]["minute"] = 0
    gaps = _uncovered(windows, schedule)
    assert gaps
    assert all(g.startswith("daily_rule_signals ") for g in gaps)


def test_moving_retry_minute_breaks_cover() -> None:
    windows = _load_windows()
    schedule = copy.deepcopy(run_scheduler.SCHEDULE_CONFIG)
    schedule["daily_settle_orders_retry"]["hour"] = 10
    gaps = _uncovered(windows, schedule)
    assert any(g.startswith("daily_settle_orders_retry ") for g in gaps)


@pytest.mark.parametrize("index", range(7))
def test_dropping_any_window_breaks_cover(index: int) -> None:
    windows = _load_windows()
    if index >= len(windows):
        pytest.skip("窓の数が少ない")
    reduced = windows[:index] + windows[index + 1 :]
    assert _uncovered(reduced, run_scheduler.SCHEDULE_CONFIG)


def test_tuesday_window_does_not_cover_wednesday() -> None:
    windows = [{"start": "00:00", "end": "02:00", "name": "x", "day": "Tuesday"}]
    assert _is_covered(windows, 1, 60)
    assert not _is_covered(windows, 2, 60)
    assert not _is_covered(windows, 1, 120)


def test_plain_window_is_weekday_only() -> None:
    windows = [{"start": "16:00", "end": "17:00", "name": "x"}]
    assert _is_covered(windows, 4, 16 * 60)
    assert not _is_covered(windows, 5, 16 * 60)
