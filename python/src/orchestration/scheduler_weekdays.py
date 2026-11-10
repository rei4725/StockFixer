"""
ジョブの設定の day_of_week（"mon-fri"・"tue"・"*" など）を曜日の番号に読む

scheduler_queue.py の行数の上限のため、SchedulerQueueManager から分けた（#1783）。
"""

from __future__ import annotations

_WEEKDAY_MAP = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}


def parse_day_of_week(expr: str) -> set[int]:
    """曜日の番号（月=0）の集まりを返す。空は毎日を表す。"""
    expr = (expr or "").strip().lower()
    if not expr or expr == "*":
        return set()

    weekdays: set[int] = set()
    for token in [part.strip() for part in expr.split(",") if part.strip()]:
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = _WEEKDAY_MAP.get(start_text)
            end = _WEEKDAY_MAP.get(end_text)
            if start is None or end is None:
                continue
            if start <= end:
                weekdays.update(range(start, end + 1))
            else:
                weekdays.update(range(start, 7))
                weekdays.update(range(0, end + 1))
        else:
            value = _WEEKDAY_MAP.get(token)
            if value is not None:
                weekdays.add(value)
    return weekdays
