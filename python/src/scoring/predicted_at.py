"""予想の時刻から、採点の基準の足の日を決める（rule_report.py の prediction_signals が使う）。

基準の日は prediction/base_bar_date.py の _base_bar_date と同じ決め方にする（#2066）。
scoring から prediction は import しないので、引けの時刻は domain.market_close から読む。
"""

from __future__ import annotations

import pandas as pd

from src.domain.market_close import MARKET_CLOSE


def day_key(s: pd.Series) -> pd.Series:
    """merge_asof の鍵にする日時（分解能をそろえる。pandas は入力ごとに単位が変わる）。"""
    return pd.to_datetime(s).astype("datetime64[ns]")


def parse_predicted_at(s: pd.Series) -> pd.Series:
    """YYYYMMDD_HHMMSS（UTC）を UTC の時刻にする。読めなければ ISO 8601 として読む。"""
    text = s.astype(str)
    ts = pd.to_datetime(text, format="%Y%m%d_%H%M%S", errors="coerce", utc=True)
    rest = ts.isna()
    if rest.any():
        ts[rest] = pd.to_datetime(text[rest], format="ISO8601", errors="coerce", utc=True)
    return ts


def base_day(ts: pd.Series, market: str) -> pd.Series:
    """予想の時刻（UTC）に引けていた最後の日を、0 時の日時で返す。

    市場の現地の時刻が引けより前なら現地の前の日、引けの後ならその日。市場が表に
    無ければ UTC の日付を返す（_base_bar_date と同じ）。
    """
    if market not in MARKET_CLOSE:
        return day_key(ts.dt.tz_localize(None).dt.normalize())
    tz, close_time = MARKET_CLOSE[market]
    local = ts.dt.tz_convert(tz)
    day = local.dt.tz_localize(None).dt.normalize()
    # 時刻が読めない行（NaT）は秒が NaN で比べると False になり、day も NaT のまま捨てられる
    seconds = local.dt.hour * 3600 + local.dt.minute * 60 + local.dt.second
    close_seconds = close_time.hour * 3600 + close_time.minute * 60 + close_time.second
    before_close = seconds < close_seconds
    return day_key(day.where(~before_close, day - pd.Timedelta(days=1)))
