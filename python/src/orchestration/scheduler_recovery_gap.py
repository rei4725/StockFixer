"""
失敗の回の後の補完に、前の回の終わりからの間を空ける判定（#1607）

recovery_min_gap_minutes を持つジョブは、同じ期間の前の回の終わりからその分だけ補完しない。
回数の上限を短い障害の間に使い切らないためである。欄の無いジョブは間を空けない。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from src.utils.japan_time import parse_iso_datetime


def _last_attempt_at(events: list[dict[str, Any]], job_id: str, period_key: str) -> datetime | None:
    """同じ期間の失敗の回の終わりと始めの記録のうち、いちばん新しい時刻を返す。

    前の起動のプロセスで消えた回には終わりの記録が無いので、始めの記録も見る。
    """
    latest: datetime | None = None
    for event in events:
        if event.get("job_id") != job_id or event.get("period_key") != period_key:
            continue
        status = event.get("status")
        if status not in {"started", "error"}:
            continue
        stamp = event.get("finished_at") if status == "error" else event.get("started_at")
        if not isinstance(stamp, str):
            continue
        try:
            attempted_at = parse_iso_datetime(stamp)
        except ValueError:
            continue
        if latest is None or attempted_at > latest:
            latest = attempted_at
    return latest


def is_within_recovery_gap(
    config: dict[str, Any],
    events: list[dict[str, Any]],
    job_id: str,
    period_key: str,
    now: datetime,
) -> bool:
    """前の回の終わりからの間が明けていなければ True。欄の無いジョブは常に False。"""
    gap_minutes = config.get("recovery_min_gap_minutes")
    if gap_minutes is None:
        return False
    last_attempt = _last_attempt_at(events, job_id, period_key)
    if last_attempt is None:
        return False
    return now < last_attempt + timedelta(minutes=gap_minutes)
