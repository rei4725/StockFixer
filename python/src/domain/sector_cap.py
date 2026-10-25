"""セクターごとの保有の上限で候補を絞る、domain の純粋なルール関数。

backtest と trading の両 BC から参照する。セクターの照会（yfinance）は src.utils.sector_constraints に残す。
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, Mapping, TypeVar

T = TypeVar("T")


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
