"""
ペーパートレードの評価額の計算が読む記録の読み出し（paper_equity.py から分けた）。

約定済み注文（DB の paper_orders）、日次終値（load_raw_ohlcv）、分割の記録（yfinance）を返す。
paper_equity は、これらを import し直して今の名前で呼ぶ。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import yfinance as yf

from src.utils.data_path_utils import get_ticker
from src.utils.db import load_raw_ohlcv
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_filled_orders() -> pd.DataFrame:
    """約定済み注文を時系列順に取得する。"""
    with _db_connection() as con:
        df = pd.read_sql(
            """
            SELECT market, symbol, side, qty, fill_price, filled_at
            FROM paper_orders
            WHERE status = 'filled' AND fill_price IS NOT NULL AND filled_at IS NOT NULL
              AND fill_price <> 'NaN'
            ORDER BY filled_at
            """,
            con,
        )
    return df


def _default_price_loader(market: str, symbol: str) -> Optional[pd.Series]:
    """銘柄の日次終値 Series を返す（index=DatetimeIndex）。"""
    df = load_raw_ohlcv(market, symbol)
    if df is None or df.empty or "Close" not in df.columns:
        return None
    series = df["Close"].copy()
    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
    return series


def _default_split_loader(market: str, symbol: str) -> Optional[pd.Series]:
    """銘柄の分割の記録 Series を yfinance から返す（index=分割の日、値=分割の比）。

    取れなければ None（分割で直さずに評価する）。
    """
    ticker = get_ticker(market, symbol)
    try:
        splits = yf.Ticker(ticker).splits
    except Exception:
        logger.warning(
            "分割の記録の取得失敗のため直さずに評価: %s/%s", market, symbol, exc_info=True
        )
        return None
    if splits is None or splits.empty:
        return None
    splits = splits.copy()
    idx = pd.to_datetime(splits.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    splits.index = idx.normalize()
    return splits
