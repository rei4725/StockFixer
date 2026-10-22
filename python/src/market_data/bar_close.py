"""
引けていない当日の足を落とす（#890）

場中や夜の補完で日次の取得が動くと、yfinance は当日の途中の足を返す。
それを確定の足として market_data_raw に残さないよう、市場の引けより前なら
その市場の当日以降の足を落とす。落とした日は DB の最新の日付にならないため、
次の回の差分取得で取り直される。
"""

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 市場ごとの (タイムゾーン, 引けの時刻)。jp は東証 15:30 JST、us は米国東部 16:00
_MARKET_CLOSE = {
    "jp": (ZoneInfo("Asia/Tokyo"), time(15, 30)),
    "us": (ZoneInfo("America/New_York"), time(16, 0)),
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def drop_unclosed_bars(df: pd.DataFrame | None, market: str) -> pd.DataFrame | None:
    """
    市場の引けより前なら、その市場の当日以降の日付の足を落とす。

    Args:
        df: 取得した OHLCV（日付の index）
        market: マーケット識別子 (例: "us", "jp")。不明な市場は落とさない

    Returns:
        引けていない足を除いた DataFrame。df が None か空ならそのまま返す。
    """
    if df is None or df.empty or market not in _MARKET_CLOSE:
        return df
    tz, close_time = _MARKET_CLOSE[market]
    local_now = _utcnow().astimezone(tz)
    if local_now.time() >= close_time:
        return df
    market_today = pd.Timestamp(local_now.date())
    index = pd.DatetimeIndex(pd.to_datetime(df.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    unclosed = index.normalize() >= market_today
    if unclosed.any():
        logger.info(
            f"引けていない足を落とす: market={market} "
            f"{market_today.strftime('%Y-%m-%d')}以降 {int(unclosed.sum())}行"
            f"（現地 {local_now.strftime('%H:%M')} < 引け {close_time.strftime('%H:%M')}）"
        )
    return df[~unclosed]
