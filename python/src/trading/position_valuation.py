"""保有ポジションの時価評価（#844）。

現在値が取れない（NaN の）保有を、総資産の計算でどう数えるかをまとめる。
"""

import logging
import math
from typing import Any, Iterable, Mapping


def positions_market_value(positions: Iterable[Mapping[str, Any]], logger: logging.Logger) -> float:
    """保有ポジションの時価評価額の合計を返す。

    評価できない保有は時価0として数える（#844）。NaN で総資産ごと NaN になると
    DD が 0.0 になりゲートが外れるため、止める側（DD を大きく見る側）に倒す。
    WARNING は呼び出し側の logger に出す。
    """
    market_value = 0.0
    for p in positions:
        price = float(p.get("current_price", 0.0))
        if math.isnan(price):
            logger.warning(f"[risk] {p.get('symbol')}: 現在値が取れず時価0として総資産に数える")
            continue
        market_value += float(p.get("qty", 0)) * price
    return market_value


def long_positions_unrealized_pnl(
    positions: Iterable[Mapping[str, Any]], logger: logging.Logger
) -> float:
    """買い持ちの保有の含み損益の合計を返す（#1162）。

    評価できない保有は空売りの short_positions_unrealized_pnl と同じく数えずに飛ばし、
    NaN で合計ごと NaN にしない。WARNING は呼び出し側の logger に出す。
    """
    total = 0.0
    for p in positions:
        pnl = float(p.get("unrealized_pnl", 0.0))
        if math.isnan(pnl):
            logger.warning(f"[risk] {p.get('symbol')}: 買い持ちの現在値が取れず含み損益を数えない")
            continue
        total += pnl
    return total


def short_positions_unrealized_pnl(
    positions: Iterable[Mapping[str, Any]], logger: logging.Logger
) -> float:
    """空売りの保有の含み損益の合計を返す（#1148）。

    空売りは建てても現金が動かないため、総資産には時価ではなく含み損益を足す。
    評価できない保有は買い持ちの positions_market_value と同じく数えずに飛ばし、
    NaN で合計ごと NaN にしない。WARNING は呼び出し側の logger に出す。
    """
    total = 0.0
    for p in positions:
        pnl = float(p.get("unrealized_pnl", 0.0))
        if math.isnan(pnl):
            logger.warning(f"[risk] {p.get('symbol')}: 空売りの現在値が取れず含み損益を数えない")
            continue
        total += pnl
    return total


def read_short_positions(broker: Any) -> list[dict[str, Any]]:
    """broker の空売りの保有を返す。get_short_positions を持たない broker は空とする。"""
    try:
        return list(broker.get_short_positions())
    except AttributeError:
        return []


def coverable_short_qty(pos: Mapping[str, Any]) -> int:
    """返済に出してよい株数（約定待ちの SHORT の pending_qty を引いた株数、#1225）。

    約定前の空売りを返すと、仮登録の単価で幻の実現損益が出て、後の約定で空売りが残る。
    """
    return int(pos.get("qty") or 0) - int(pos.get("pending_qty") or 0)


def count_short_positions(broker: Any) -> int:
    """保有の上限に数える空売りの建玉の数（qty > 0）を返す（#1146）。"""
    return sum(1 for p in read_short_positions(broker) if p.get("qty", 0) > 0)


def short_held_codes(broker: Any) -> set[str]:
    """空売りの建玉（qty > 0。約定待ちの SHORT の仮登録を含む）の銘柄コードを返す（#1349）。"""
    return {
        str(p["symbol"]).replace(".T", "")
        for p in read_short_positions(broker)
        if p.get("qty", 0) > 0
    }


def broker_positions_value(broker: Any, logger: logging.Logger) -> float:
    """broker の保有を総資産に足す額（買い持ちの時価 + 空売りの含み損益）を返す（#1148）。

    get_short_positions を持たない broker（AttributeError）は空売りを空として扱う。
    """
    market_value = positions_market_value(broker.get_positions(), logger)
    short_positions = read_short_positions(broker)
    return market_value + short_positions_unrealized_pnl(short_positions, logger)
