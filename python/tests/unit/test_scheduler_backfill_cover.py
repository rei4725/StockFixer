"""scheduler_backfill_cover.has_success_since を events の並びだけで直に確かめる（#1854）。"""

from __future__ import annotations

from typing import Any

from src.orchestration.scheduler_backfill_cover import has_success_since

JOB = "daily_job"
KEY = "2026-11-15"


def _event(key: Any, *, status: str = "success", job_id: str = JOB) -> dict[str, Any]:
    return {"job_id": job_id, "period_key": key, "status": status}


def test_no_events_is_false() -> None:
    assert has_success_since([], JOB, KEY) is False


def test_same_day_success_is_true() -> None:
    assert has_success_since([_event(KEY)], JOB, KEY) is True


def test_later_day_success_is_true() -> None:
    assert has_success_since([_event("2026-11-16")], JOB, KEY) is True


def test_earlier_day_success_is_false() -> None:
    assert has_success_since([_event("2026-11-14")], JOB, KEY) is False


def test_other_job_success_is_false() -> None:
    assert has_success_since([_event(KEY, job_id="other_job")], JOB, KEY) is False


def test_error_is_false() -> None:
    assert has_success_since([_event(KEY, status="error")], JOB, KEY) is False


def test_started_is_false() -> None:
    assert has_success_since([_event("2026-11-16", status="started")], JOB, KEY) is False


def test_week_key_is_not_compared_with_day_key() -> None:
    # 週の鍵は日の鍵と長さが違い、文字では "2026-W47" > "2026-11-15" でも比べない
    assert "2026-W47" > KEY
    assert has_success_since([_event("2026-W47")], JOB, KEY) is False


def test_day_key_is_not_compared_with_week_key() -> None:
    assert has_success_since([_event("2026-11-16")], JOB, "2026-W46") is False


def test_non_string_key_is_skipped() -> None:
    assert has_success_since([_event(None), _event(20261116)], JOB, KEY) is False


def test_one_match_among_noise_is_true() -> None:
    events = [
        _event("2026-W47"),
        _event(KEY, status="error"),
        _event(KEY, job_id="other_job"),
        _event("2026-11-16"),
    ]
    assert has_success_since(events, JOB, KEY) is True
