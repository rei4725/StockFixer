from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Callable, Iterable, Mapping, TypeVar

import yfinance as yf

from src.utils.data_path_utils import get_ticker
from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def _normalize_sector(sector: str | None, fallback_key: str) -> str:
    if sector and str(sector).strip():
        return str(sector).strip()
    return f"UNKNOWN:{fallback_key}"


class _SectorUnresolvedError(Exception):
    """セクターが引けなかったことを示す。lru_cache に結果を残さないために使う。"""


@lru_cache(maxsize=2048)
def _lookup_symbol_sector(market: str, symbol: str) -> str:
    """引けたセクターだけをキャッシュする。引けないときは例外で抜け、次回引き直す。"""
    ticker = get_ticker(market, symbol)
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        logger.warning(
            "[sector] sector lookup failed: %s/%s: %s", market, symbol, exc, exc_info=True
        )
        raise _SectorUnresolvedError from exc

    sector = info.get("sector") or info.get("industry")
    if not (sector and str(sector).strip()):
        raise _SectorUnresolvedError
    return _normalize_sector(sector, f"{market}:{symbol}")


def get_symbol_sector(market: str, symbol: str) -> str:
    """銘柄のセクター情報を yfinance から取得し、引けたものだけをプロセス内でキャッシュする。

    yfinance の例外やセクターが空のときは UNKNOWN を返すが、キャッシュせず次回引き直す。
    """
    try:
        return _lookup_symbol_sector(market, symbol)
    except _SectorUnresolvedError:
        return _normalize_sector(None, f"{market}:{symbol}")


get_symbol_sector.cache_clear = _lookup_symbol_sector.cache_clear  # type: ignore[attr-defined]


def filter_by_sector_cap(
    items: Iterable[T],
    max_sector_positions: int,
    sector_getter: Callable[[T], str],
    initial_counts: Mapping[str, int] | None = None,
) -> list[T]:
    """セクターごとの上限件数を超えないように順序を保ってフィルタする。

    initial_counts を渡すと、その件数（保有中の銘柄の業種の数など）から数え始める。
    """
    if max_sector_positions <= 0:
        return list(items)

    selected: list[T] = []
    sector_counts: Counter[str] = Counter(initial_counts or {})
    for item in items:
        sector = sector_getter(item)
        if sector_counts[sector] >= max_sector_positions:
            continue
        selected.append(item)
        sector_counts[sector] += 1
    return selected
