"""allocation_rebalance_log テーブルの読み書き。"""

from typing import Optional

from src.trading.allocation_strategy.types import AllocationSnapshot
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_latest_snapshot() -> Optional[AllocationSnapshot]:
    """最新の状態行を返す。まだ1行も無ければ None。"""
    with _db_connection() as con:
        row = con.execute("""
            SELECT id, executed_at, action, tqqq_price, shy_price,
                   tqqq_qty_before, shy_qty_before, cash_before,
                   tqqq_qty_after, shy_qty_after, cash_after
            FROM allocation_rebalance_log
            ORDER BY id DESC
            LIMIT 1
            """).fetchone()
    if row is None:
        return None
    return AllocationSnapshot(
        id=row[0],
        executed_at=row[1],
        action=row[2],
        tqqq_price=row[3],
        shy_price=row[4],
        tqqq_qty_before=row[5],
        shy_qty_before=row[6],
        cash_before=row[7],
        tqqq_qty_after=row[8],
        shy_qty_after=row[9],
        cash_after=row[10],
    )


def insert_snapshot(
    action: str,
    tqqq_price: float,
    shy_price: float,
    tqqq_qty_before: float,
    shy_qty_before: float,
    cash_before: float,
    tqqq_qty_after: float,
    shy_qty_after: float,
    cash_after: float,
) -> None:
    """新しい状態行を追記する。"""
    with _db_connection() as con:
        con.execute(
            """
            INSERT INTO allocation_rebalance_log (
                action, tqqq_price, shy_price,
                tqqq_qty_before, shy_qty_before, cash_before,
                tqqq_qty_after, shy_qty_after, cash_after
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                action,
                tqqq_price,
                shy_price,
                tqqq_qty_before,
                shy_qty_before,
                cash_before,
                tqqq_qty_after,
                shy_qty_after,
                cash_after,
            ],
        )
    logger.info(
        "allocation_rebalance_log 追記: action=%s tqqq_qty=%.4f shy_qty=%.4f cash=%.2f",
        action,
        tqqq_qty_after,
        shy_qty_after,
        cash_after,
    )
