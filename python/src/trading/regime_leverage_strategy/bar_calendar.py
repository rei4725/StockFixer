"""レジームレバレッジ戦略の、米国の営業日と日足の引けの時刻を数える純な関数(#2181)。

どれも now・since を引数で受け、datetime.now() を呼ばず、DB もロガーも使わない。
service.py から中身を変えずに移した。
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from src.domain.market_close import MARKET_CLOSE
from src.domain.us_market_holidays import is_us_business_day

# 米国の引けのタイムゾーンと時刻は表から読む（#2121）
_US_MARKET_TZ: ZoneInfo = MARKET_CLOSE["us"][0]
_US_MARKET_CLOSE: time = MARKET_CLOSE["us"][1]


def _as_aware(dt: datetime) -> datetime:
    """タイムゾーンの無い時刻は UTC とみなす(DB の executed_at・entry_date はセッションの
    タイムゾーン UTC で、タイムゾーンの無い時刻として入る。#942)。"""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _bar_close_at(bar_date: object) -> datetime:
    """日足の引けの時刻(米国の取引所の16:00、夏時間・冬時間は ZoneInfo が扱う)を返す。"""
    day = pd.Timestamp(bar_date).date()
    return datetime.combine(day, _US_MARKET_CLOSE, tzinfo=_US_MARKET_TZ)


def _latest_expected_bar_date(now: datetime) -> date:
    """now までに引けたはずの一番新しい米国の営業日を返す(金曜が休場の週は木曜。#1608)。"""
    now_aware = _as_aware(now)
    day = now_aware.astimezone(_US_MARKET_TZ).date()
    while not is_us_business_day(day) or _bar_close_at(day) > now_aware:
        day -= timedelta(days=1)
    return day


def _last_bar_date(df: pd.DataFrame) -> date:
    return pd.Timestamp(df.index[-1]).date()


def _us_business_days_behind(last_bar: date, expected_bar: date) -> int:
    """last_bar の翌日から expected_bar までの米国の営業日の数を返す。"""
    calendar_days = (expected_bar - last_bar).days
    return sum(
        is_us_business_day(last_bar + timedelta(days=n)) for n in range(1, calendar_days + 1)
    )


def _missing_business_days(df: pd.DataFrame, since: datetime, last_bar: date) -> list[date]:
    """since の後に引けたはずで last_bar より前の米国の営業日のうち、df に足が無い日を返す。"""
    since_aware = _as_aware(since)
    first_day = since_aware.astimezone(_US_MARKET_TZ).date()
    bar_days = {pd.Timestamp(ts).date() for ts in df.index}
    days = (first_day + timedelta(days=n) for n in range((last_bar - first_day).days))
    return [
        day
        for day in days
        if is_us_business_day(day) and _bar_close_at(day) > since_aware and day not in bar_days
    ]


def select_unchecked_closed_bars(
    df: pd.DataFrame, last_checked_at: Optional[datetime], now: datetime
) -> pd.DataFrame:
    """日次判定で見る日足を返す: now までに引けた足のうち、前の回(last_checked_at)の後に
    引けた足を全部。引けていない当日の足は含めない。前の回の後に引けた足が無ければ、
    最後に引けた足だけを返す(#891)。
    """
    now_aware = _as_aware(now)
    closes = [_bar_close_at(ts) for ts in df.index]
    closed = df[[close_at <= now_aware for close_at in closes]]
    if closed.empty:
        return closed
    if last_checked_at is not None:
        since = _as_aware(last_checked_at)
        unchecked = closed[[_bar_close_at(ts) > since for ts in closed.index]]
        if not unchecked.empty:
            return unchecked
    return closed.iloc[[-1]]


def _weekdays_behind(last_bar: date, expected_bar: date) -> int:
    """last_bar の翌日から expected_bar までの平日の数を返す(週末は数えない)。"""
    return len(pd.bdate_range(last_bar + timedelta(days=1), expected_bar))
