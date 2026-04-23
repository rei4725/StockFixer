from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Callable, Iterable, TypeVar

import yfinance as yf

from src.utils.data_path_utils import get_ticker
from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def _normalize_sector(sector: str | None, fallback_key: str) -> str:
    if sector and str(sector).strip():
        return str(sector).strip()
    return f"UNKNOWN:{fallback_key}"


@lru_cache(maxsize=2048)
def get_symbol_sector(market: str, symbol: str) -> str:
    """銘柄のセクター情報を yfinance から取得し、プロセス内でキャッシュする。"""
    ticker = get_ticker(market, symbol)
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        logger.warning(
            "[sector] sector lookup failed: %s/%s: %s", market, symbol, exc, exc_info=True
        )
        return _normalize_sector(None, f"{market}:{symbol}")

    sector = info.get("sector") or info.get("industry")
    return _normalize_sector(sector, f"{market}:{symbol}")


def filter_by_sector_cap(
    items: Iterable[T],
    max_sector_positions: int,
    sector_getter: Callable[[T], str],
) -> list[T]:
    """セクターごとの上限件数を超えないように順序を保ってフィルタする。"""
    if max_sector_positions <= 0:
        return list(items)

    selected: list[T] = []
    sector_counts: Counter[str] = Counter()
    for item in items:
        sector = sector_getter(item)
        if sector_counts[sector] >= max_sector_positions:
            continue
        selected.append(item)
        sector_counts[sector] += 1
    return selected
