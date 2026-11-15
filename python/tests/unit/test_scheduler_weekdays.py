"""parse_day_of_week の読み方と、SCHEDULE_CONFIG の曜日の綴りの見張り（#1834）

空の集まりは「毎日」を表すため、綴りの誤りは補完の判定だけを毎日に化けさせる。
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import run_scheduler

from src.orchestration.scheduler_weekdays import parse_day_of_week


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("*", set()),
        ("", set()),
        ("   ", set()),
        ("mon-fri", {0, 1, 2, 3, 4}),
        ("mon-sun", {0, 1, 2, 3, 4, 5, 6}),
        ("fri-mon", {4, 5, 6, 0}),
        ("sun-tue", {6, 0, 1}),
        ("mon,wed", {0, 2}),
        (" mon , wed ", {0, 2}),
        ("tue", {1}),
        ("sat", {5}),
        ("sun", {6}),
        ("MON-FRI", {0, 1, 2, 3, 4}),
        ("Tue", {1}),
        ("mon-fri,sun", {0, 1, 2, 3, 4, 6}),
        ("mon,xyz", {0}),
        ("mon-fir,sat", {5}),
    ],
)
def test_parse_day_of_week_table(expr: str, expected: set[int]) -> None:
    assert parse_day_of_week(expr) == expected


@pytest.mark.parametrize("expr", ["mon-fir", "xyz", "monday", "1-5", "mon-", "-fri"])
def test_unreadable_spelling_becomes_empty(expr: str) -> None:
    """読めない綴りは空、つまり「毎日」に化ける。見張りのテストが拾うべき形。"""
    assert parse_day_of_week(expr) == set()


def test_none_is_treated_as_every_day() -> None:
    assert parse_day_of_week(None) == set()  # type: ignore[arg-type]


def _unreadable_jobs(config: Mapping[str, Mapping[str, object]]) -> list[str]:
    bad: list[str] = []
    for name, job in config.items():
        expr = str(job.get("day_of_week", "*")).strip()
        if expr != "*" and not parse_day_of_week(expr):
            bad.append(f"{name}: {expr!r}")
    return bad


def test_schedule_config_day_of_week_all_readable() -> None:
    """* 以外の day_of_week は、すべて空でない曜日の集まりに読める。"""
    assert run_scheduler.SCHEDULE_CONFIG
    assert _unreadable_jobs(run_scheduler.SCHEDULE_CONFIG) == []


def test_guard_catches_misspelled_day_of_week() -> None:
    """見張りが空振りしないこと: 綴りを1つ壊すと当たる。"""
    config = {name: dict(job) for name, job in run_scheduler.SCHEDULE_CONFIG.items()}
    first = next(iter(config))
    config[first]["day_of_week"] = "mon-fir"
    assert _unreadable_jobs(config) == [f"{first}: 'mon-fir'"]
