from __future__ import annotations

from functools import lru_cache

import yfinance as yf

from src.domain.sector_cap import filter_by_sector_cap as filter_by_sector_cap  # noqa: F401
from src.utils.data_path_utils import get_ticker
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 業種を引けなかった銘柄は全部この1つの業種として数え、同一業種の上限を当てる（#1469）。
UNKNOWN_SECTOR = "UNKNOWN"


def _normalize_sector(sector: str | None) -> str:
    if sector and str(sector).strip():
        return str(sector).strip()
    return UNKNOWN_SECTOR


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
    return _normalize_sector(sector)


def get_symbol_sector(market: str, symbol: str) -> str:
    """銘柄のセクター情報を yfinance から取得し、引けたものだけをプロセス内でキャッシュする。

    yfinance の例外やセクターが空のときは、銘柄によらず同じ UNKNOWN_SECTOR を返す。
    引けない銘柄どうしも同一業種の上限に当たる（#1469）。キャッシュはせず次回引き直す。
    """
    try:
        return _lookup_symbol_sector(market, symbol)
    except _SectorUnresolvedError:
        return UNKNOWN_SECTOR


get_symbol_sector.cache_clear = _lookup_symbol_sector.cache_clear  # type: ignore[attr-defined]
