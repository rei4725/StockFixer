"""夜間の仮の入れ替えが特徴量の待ちで飛んだ夜の原因の分け方（#2086）。

stock_features は東証の営業日の朝の日次の道でしか作り直さない。印を書いた後に特徴量の
コードが変わっても、次の営業日の朝の日次の道が来るまでは印が古いのが当たり前である。

- WAIT_NEXT_DAILY_RUN: 印を書いた後に営業日の朝の日次の道の時刻がまだ来ていない
  （週末・祝日・その日の昼のデプロイ）。故障ではないので晩に数えない
- DAILY_RUN_WITHOUT_MARKER: 印を書いた後に日次の道の時刻を過ぎたのに印が今の版でない
  （取得の止まり・作れなかった銘柄の線）。故障として晩に数える

印が無い・書いた時刻が読めないときは、今までどおり故障の側に倒す。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from src.domain.jp_market_holidays import is_jp_business_day
from src.utils.japan_time import to_jst

WAIT_NEXT_DAILY_RUN = "wait_next_daily_run"
DAILY_RUN_WITHOUT_MARKER = "daily_run_without_marker"

# 日次の道の時刻（run_scheduler.py の daily_pipeline の hour・minute と同じ）。
# 揃っていることは tests/unit/test_daily_run_time_matches_schedule.py が確かめる（#2088）
DAILY_RUN_TIME = time(7, 30)
# 次の予定を探す日数の上限（年末年始の休場が続いても届く長さ）
_MAX_LOOKAHEAD_DAYS = 31


@dataclass(frozen=True)
class FeaturesWait:
    """特徴量の待ちの原因と、次の日次の道の予定（JST）。"""

    cause: str
    next_daily_run: datetime


def _daily_run_at(day: date, tzinfo) -> datetime:
    return datetime.combine(day, DAILY_RUN_TIME, tzinfo=tzinfo)


def next_daily_run(now: datetime) -> datetime:
    """now より後の、最初の東証の営業日の日次の道の時刻（JST）。"""
    now_j = to_jst(now)
    day = now_j.date()
    for _ in range(_MAX_LOOKAHEAD_DAYS):
        run_at = _daily_run_at(day, now_j.tzinfo)
        if is_jp_business_day(day) and run_at > now_j:
            return run_at
        day += timedelta(days=1)
    return _daily_run_at(day, now_j.tzinfo)


def classify_features_wait(written_at: datetime | None, now: datetime) -> FeaturesWait:
    """印を書いた時刻と今から、特徴量の待ちの原因を分ける。

    印を書いた後、今までに東証の営業日の日次の道の時刻があれば DAILY_RUN_WITHOUT_MARKER、
    無ければ WAIT_NEXT_DAILY_RUN。
    """
    upcoming = next_daily_run(now)
    if written_at is None:
        return FeaturesWait(DAILY_RUN_WITHOUT_MARKER, upcoming)
    first_after_marker = next_daily_run(written_at)
    cause = WAIT_NEXT_DAILY_RUN if first_after_marker > to_jst(now) else DAILY_RUN_WITHOUT_MARKER
    return FeaturesWait(cause, upcoming)


__all__ = [
    "DAILY_RUN_TIME",
    "DAILY_RUN_WITHOUT_MARKER",
    "WAIT_NEXT_DAILY_RUN",
    "FeaturesWait",
    "classify_features_wait",
    "next_daily_run",
]
