"""
決算・イベントカレンダーヘルパー (I-256)

yfinance Ticker.calendar / get_earnings_dates からイベント日を取得し、
DuckDB にキャッシュして API 呼び出しを最小化する。
"""

from __future__ import annotations

from typing import Any, Optional, Union

import pandas as pd
import yfinance as yf

from src.utils.data_path_utils import get_ticker
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CACHE_TTL_DAYS = 7


def get_event_dates(market: str, symbol: str) -> pd.DatetimeIndex:
    """
    決算・イベント日を返す（DuckDB キャッシュ優先）。

    キャッシュが TTL 内であれば DB から返す。
    TTL 切れまたは未キャッシュの場合は yfinance から再取得して DB に保存する。
    取得失敗時は空の DatetimeIndex を返す。
    """
    cached = _load_cached_event_dates(market, symbol)
    if cached is not None:
        return cached
    dates = _fetch_event_dates_from_yfinance(market, symbol)
    _save_event_dates(market, symbol, dates)
    return dates


def is_near_event(
    check_date: Union[str, Any],
    market: str,
    symbol: str,
    days_before: int = 2,
    days_after: int = 1,
    event_dates: Optional[pd.DatetimeIndex] = None,
) -> bool:
    """
    指定日がイベント日の前後 days_before/days_after 日以内かどうかを判定する。

    Args:
        check_date: 判定する日付（date / datetime / str / Timestamp）
        market: マーケット識別子
        symbol: 銘柄シンボル
        days_before: イベント日の何日前からフィルタするか（デフォルト 2）
        days_after: イベント日の何日後までフィルタするか（デフォルト 1）
        event_dates: 事前取得済みのイベント日一覧（省略時は get_event_dates() で取得）

    Returns:
        True なら check_date はイベント近傍
    """
    if event_dates is None:
        event_dates = get_event_dates(market, symbol)
    if len(event_dates) == 0:
        return False
    check_ts = pd.Timestamp(check_date).normalize()
    for ev in event_dates:
        ev_ts = pd.Timestamp(ev).normalize()
        delta = (check_ts - ev_ts).days
        if -days_before <= delta <= days_after:
            return True
    return False


def _load_cached_event_dates(market: str, symbol: str) -> Optional[pd.DatetimeIndex]:
    """DuckDB キャッシュからイベント日を取得する。TTL 切れ・未登録の場合は None を返す。"""
    try:
        with _db_connection() as con:
            rows = con.execute(
                """
                SELECT event_date, fetched_at
                FROM earnings_calendar
                WHERE market = ? AND symbol = ?
                ORDER BY event_date
                """,
                [market, symbol],
            ).fetchall()
        if not rows:
            return None
        latest_fetch = max(pd.Timestamp(r[1]) for r in rows)
        if (pd.Timestamp.now() - latest_fetch).days > _CACHE_TTL_DAYS:
            return None
        dates = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])
        return dates
    except Exception as exc:
        logger.warning(
            "イベントカレンダーキャッシュ読込失敗 [%s/%s]: %s", market, symbol, exc, exc_info=True
        )
        return None


def _fetch_event_dates_from_yfinance(market: str, symbol: str) -> pd.DatetimeIndex:
    """yfinance から決算・イベント日を取得する。取得失敗時は空の DatetimeIndex を返す。"""
    ticker = get_ticker(market, symbol)
    dates: set[pd.Timestamp] = set()
    try:
        ticker_obj = yf.Ticker(ticker)
        # Ticker.calendar: 直近の予定イベント（日本株・米国株とも対応）
        try:
            cal = ticker_obj.calendar
            if cal is not None:
                _extract_calendar_dates(cal, dates)
        except Exception:
            logger.debug("calendar 取得スキップ [%s]", ticker, exc_info=True)
        # get_earnings_dates: 過去・将来の決算日リスト
        try:
            getter = getattr(ticker_obj, "get_earnings_dates", None)
            if callable(getter):
                earnings_df = getter(limit=12)
            else:
                earnings_df = getattr(ticker_obj, "earnings_dates", pd.DataFrame())
            if earnings_df is not None and len(earnings_df) > 0:
                raw = earnings_df.index if isinstance(earnings_df, pd.DataFrame) else earnings_df
                for d in pd.DatetimeIndex(pd.to_datetime(raw, errors="coerce")).dropna():
                    dates.add(d.normalize())
        except Exception:
            logger.debug("earnings_dates 取得スキップ [%s]", ticker, exc_info=True)
    except Exception as exc:
        logger.warning("イベント日取得失敗 [%s/%s]: %s", market, symbol, exc, exc_info=True)
    if not dates:
        return pd.DatetimeIndex([])
    result = pd.DatetimeIndex(sorted(dates))
    if result.tz is not None:
        result = result.tz_localize(None)
    return result


def _extract_calendar_dates(
    cal: Union[dict[str, Any], pd.DataFrame], dates: set[pd.Timestamp]
) -> None:
    """calendar オブジェクト（dict または DataFrame）から日付を抽出して dates に追加する。"""
    if isinstance(cal, dict):
        for key in ("Earnings Date", "earnings_date"):
            if key not in cal:
                continue
            v = cal[key]
            candidates = v if isinstance(v, (list, tuple)) else [v]
            for d in candidates:
                try:
                    ts = pd.Timestamp(d)
                    if not pd.isna(ts):
                        dates.add(ts.normalize())
                except Exception:
                    pass
    elif isinstance(cal, pd.DataFrame):
        for key in ("Earnings Date", "earnings_date"):
            if key not in cal.index:
                continue
            v = cal.loc[key]
            vals = list(v.values) if hasattr(v, "values") else [v]
            for d in vals:
                try:
                    ts = pd.Timestamp(d)
                    if not pd.isna(ts):
                        dates.add(ts.normalize())
                except Exception:
                    pass


def _save_event_dates(market: str, symbol: str, dates: pd.DatetimeIndex) -> None:
    """イベント日を DuckDB に保存する（既存レコードは fetched_at を更新）。"""
    if len(dates) == 0:
        return
    rows = [
        {
            "market": market,
            "symbol": symbol,
            "event_date": pd.Timestamp(d).strftime("%Y-%m-%d"),
            "event_type": "earnings",
        }
        for d in dates
    ]
    df = pd.DataFrame(rows)
    try:
        with _db_connection() as con:
            con.register("_ec_temp", df)
            con.execute("""
                INSERT OR REPLACE INTO earnings_calendar
                    (market, symbol, event_date, event_type, fetched_at)
                SELECT market, symbol, event_date, event_type, CURRENT_TIMESTAMP
                FROM _ec_temp
                """)
    except Exception as exc:
        logger.warning("イベント日保存失敗 [%s/%s]: %s", market, symbol, exc, exc_info=True)
