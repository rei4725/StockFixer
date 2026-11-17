"""朝の目線の告知に使う指数・先物・為替の直近2本の終値の取得（#1875 段3）。

yf_client で 1 銘柄ずつ日足を取り、最後の足と 1 つ前の足の終値を返す。
取れなかった銘柄は欠けとして返し、残りの銘柄の取得は続ける。
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from src.market_data import yf_client
from src.utils.logger import get_logger

logger = get_logger(__name__)

QUOTE_PERIOD = "10d"


@dataclass(frozen=True)
class CloseQuote:
    """最後の足の終値（last）と 1 つ前の足の終値（prev_close）、最後の足の日。"""

    symbol: str
    last: float
    prev_close: float
    last_day: date


def _to_quote(symbol: str, df: pd.DataFrame) -> CloseQuote | None:
    if df.empty or "Close" not in df.columns:
        return None
    closes = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(closes) < 2:
        return None
    last_day = pd.Timestamp(closes.index[-1]).date()
    return CloseQuote(
        symbol=symbol,
        last=float(closes.iloc[-1]),
        prev_close=float(closes.iloc[-2]),
        last_day=last_day,
    )


def fetch_close_quotes(
    symbols: Sequence[str],
    download: Callable[..., pd.DataFrame] | None = None,
) -> tuple[list[CloseQuote], list[str]]:
    """(取れた銘柄の並び, 欠けた銘柄の並び)。並びは symbols の順のまま。"""
    fetch = download if download is not None else yf_client.download
    quotes: list[CloseQuote] = []
    missing: list[str] = []
    for symbol in symbols:
        try:
            quote = _to_quote(symbol, fetch(symbol, period=QUOTE_PERIOD))
        except Exception:
            logger.warning("朝の目線: %s の取得に失敗", symbol, exc_info=True)
            quote = None
        if quote is None:
            missing.append(symbol)
        else:
            quotes.append(quote)
    return quotes, missing
