"""区切りの時刻からの決済の損益を新しい順に読み、直近の連続損失回数を数える（#1311）

区切りは日次損失と同じ daily_loss_window_start() を呼び出し側が渡す。
"""

from datetime import datetime
from typing import Any

from src.trading.brokers.base import OrderSide


def recent_paper_closed_pnls(con: Any, window_start: datetime, limit: int) -> list[tuple]:
    """paper_orders の区切りからの決済（SELL・SHORT_COVER）の損益を新しい順に返す"""
    return list(
        con.execute(
            """
            SELECT realized_pnl
            FROM paper_orders
            WHERE status = 'filled'
              AND side IN (%s, %s)
              AND realized_pnl IS NOT NULL
              AND filled_at IS NOT NULL
              AND filled_at >= %s
            ORDER BY filled_at DESC
            LIMIT %s
            """,
            [int(OrderSide.SELL), int(OrderSide.SHORT_COVER), window_start, limit],
        ).fetchall()
    )


def recent_trade_pnl_closed_pnls(
    con: Any, broker_name: str, window_start: datetime, limit: int
) -> list[tuple]:
    """trade_pnl の区切りからの決済の損益を新しい順に返す"""
    return list(
        con.execute(
            """
            SELECT realized_pnl
            FROM trade_pnl
            WHERE broker = %s
              AND closed_at >= %s
            ORDER BY closed_at DESC
            LIMIT %s
            """,
            [broker_name, window_start, limit],
        ).fetchall()
    )


def count_leading_losses(rows: list[tuple]) -> int:
    """新しい順の損益から、勝ち（0 以上）が出るまでの負けの数を返す"""
    count = 0
    for (pnl,) in rows:
        if pnl < 0:
            count += 1
        else:
            break
    return count
