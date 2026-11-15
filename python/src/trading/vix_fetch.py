"""発注の時点で分かっている最新の VIX の終値を取る（R-406・#823・#1508）。

RiskManager の VIX による縮小が読む値を、yfinance と stock_features から取る。
risk_manager.py から分けた（#1851）。risk_manager は fetch_latest_vix を同じ名前で re-export する。
"""

from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

# 分ける前と同じロガーの名前で出し、本番のログの行とログを確かめるテストを変えない（#1851）
logger = get_logger("src.trading.risk_manager")

VIX_TICKER = "^VIX"
VIX_STALE_BUSINESS_DAYS = 5  # 返す VIX の日付がこれより古い営業日数なら WARNING（#823）


def _fetch_raw_vix_close() -> Optional[tuple[float, date]]:
    """yfinance から ^VIX の最新の終値とその日付を返す（#823。DB は VIX の生の値を持たない）。"""
    hist = yf.Ticker(VIX_TICKER).history(period="10d", auto_adjust=False, timeout=10)
    close = hist["Close"].dropna() if hist is not None and "Close" in hist.columns else None
    if close is None or close.empty:
        return None
    return float(close.iloc[-1]), pd.Timestamp(close.index[-1]).date()


def _fetch_lag1_vix() -> Optional[tuple[float, date]]:
    """stock_features の最新の行の vix_close_lag1（T-1 の値。#697 で生の列は無い）と行の日付。

    row_num は symbol ごとに採番され直すため date で降順ソートして銘柄横断の最新を取る。
    """
    with _db_connection() as con:
        row = con.execute("""
            SELECT vix_close_lag1, date
            FROM stock_features
            WHERE vix_close_lag1 IS NOT NULL
            ORDER BY date DESC
            LIMIT 1
            """).fetchone()
    if row is None:
        return None
    return float(row[0]), pd.Timestamp(row[1]).date()


def fetch_latest_vix() -> Optional[float]:
    """発注の時点で分かっている最新の VIX の終値（T）を返す（R-406・#823）。

    取れなければ vix_close_lag1（T-1）に WARNING つきで戻す。値の日付をログに出す。
    どちらも取れないか、VIX_STALE_BUSINESS_DAYS 営業日より古ければ、WARNING を出して
    「分からない」として None を返す（#1508。呼ぶ側は急騰と同じく縮小する）。
    """
    result: Optional[tuple[float, date]] = None
    source = VIX_TICKER
    try:
        result = _fetch_raw_vix_close()
    except Exception:
        logger.warning("[risk] VIX の最新の終値の取得に失敗しました", exc_info=True)
    if result is None:
        source = "stock_features.vix_close_lag1"
        logger.warning("[risk] VIX の最新の終値が取れないため 1営業日前の lag1 で代わりにします")
        try:
            result = _fetch_lag1_vix()
        except Exception:
            logger.warning("[risk] VIX の取得に失敗しました", exc_info=True)
    if result is None:
        logger.warning("[risk] VIX が取れないため、分からないとして縮小を掛けます")
        return None
    vix, as_of = result
    logger.info(f"[risk] VIX={vix:.2f} as_of={as_of} source={source}")
    age = int(np.busday_count(as_of, datetime.now(timezone.utc).date()))
    if age > VIX_STALE_BUSINESS_DAYS:
        logger.warning(
            f"[risk] VIX の値が古すぎるため、分からないとして縮小を掛けます "
            f"as_of={as_of} ({age}営業日前) source={source}"
        )
        return None
    return vix
