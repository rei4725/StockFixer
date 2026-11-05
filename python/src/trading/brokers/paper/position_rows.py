"""ペーパーの保有の1行を、評価を付けた形にする（#1580）

1銘柄の行の処理（分割の直し・現在値の読み）が例外を投げても、一覧ごと落とさない。落ちた銘柄は
行を捨てず、分かっている株数と単価に NaN の現在値を付けて返す（受け手は評価できない保有として扱う）。
保有の一覧が落ちると、ほかの銘柄の SL/TP・予想の売り・ホライズン決済・ルールの SELL まで止まる。
"""

import math
from typing import Any, Callable

from src.domain.ports import MarketDataPort
from src.trading.brokers.paper import split_adjust as sa
from src.trading.brokers.paper.pending_short import pending_in_row_units, split_filled_short
from src.utils.logger import get_logger

logger = get_logger(__name__)


def long_position_row(
    market_data: MarketDataPort, latest_close: Callable[[str], float], row: tuple[Any, ...]
) -> dict[str, Any]:
    """paper_positions の1行 (symbol, qty, avg_price, updated_at) を評価を付けた保有にする"""
    sym, qty, avg, updated_at = row
    try:
        qty, avg, split_known = sa.adjust_position_for_splits(
            market_data, sym, qty, avg, updated_at
        )
        current_price = latest_close(sym)
        if not split_known and sa.looks_like_split_gap(avg, current_price):
            logger.warning(f"[paper] {sym}: 分割の記録が取れず、買値と現在値の比が分割の比ほど違う")
            current_price = math.nan
        unrealized_pnl = (current_price - avg) * qty
    except Exception:
        # qty と avg は、分割の直しが通っていれば直した値（DB に書き戻した値）のまま
        logger.error(f"[paper] {sym}: 保有の行の処理に失敗（評価できない）", exc_info=True)
        current_price = unrealized_pnl = math.nan
    return {
        "symbol": sym,
        "qty": qty,
        "avg_price": avg,
        "current_price": current_price,
        "unrealized_pnl": unrealized_pnl,
    }


def short_position_row(
    market_data: MarketDataPort,
    latest_close: Callable[[str], float],
    row: tuple[Any, ...],
    orders: list[tuple[int, float, Any]] | None,
) -> dict[str, Any]:
    """paper_short_positions の1行を評価を付けた保有にする

    row は (symbol, qty, avg_short_price, updated_at)、orders はその銘柄の約定待ちの SHORT。
    落ちた銘柄は約定待ちの分が分からないので、全部を約定待ちとして返済に出さない（#1225）。
    """
    sym, raw_qty, avg_short, updated_at = row
    qty = raw_qty
    try:
        qty, avg_short, split_known = sa.adjust_short_position_for_splits(
            market_data, sym, raw_qty, avg_short, updated_at
        )
        splits = sa.fetch_splits(market_data, sym) if orders else None
        pending_qty, filled_qty, avg_short = split_filled_short(
            pending_in_row_units(orders, splits, updated_at), int(raw_qty), qty, avg_short
        )
        current_price = latest_close(sym)
        if filled_qty > 0 and not split_known and sa.looks_like_split_gap(avg_short, current_price):
            logger.warning(
                f"[paper] {sym}: 分割の記録が取れず、売り建ての単価と現在値の比が分割の比ほど違う"
            )
            current_price = math.nan
        # 空売りの未実現損益: 売り単価 - 現在価格が正なら利益。約定待ちの分は評価しない
        unrealized_pnl = (avg_short - current_price) * filled_qty if filled_qty > 0 else 0.0
    except Exception:
        # qty と avg_short は、分割の直しが通っていれば直した値（DB に書き戻した値）のまま
        logger.error(f"[paper] {sym}: 空売りの行の処理に失敗（評価できない）", exc_info=True)
        pending_qty = qty
        current_price = unrealized_pnl = math.nan
    return {
        "symbol": sym,
        "qty": qty,
        "pending_qty": pending_qty,
        "avg_short_price": avg_short,
        "current_price": current_price,
        "unrealized_pnl": unrealized_pnl,
    }
