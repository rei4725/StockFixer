"""regime_leverage_log テーブルの読み書き。"""

from typing import Optional

from src.trading.regime_leverage_strategy.types import (
    RegimeLeverageDecision,
    RegimeLeverageSnapshot,
)
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_latest_snapshot() -> Optional[RegimeLeverageSnapshot]:
    """最新の状態行を返す。まだ1行も無ければ None。"""
    with _db_connection() as con:
        row = con.execute("""
            SELECT id, executed_at, action, reason, spy_price_usd, usdjpy_rate, shares,
                   entry_date, entry_price_jpy, entry_commission_jpy, equity_at_entry_jpy,
                   stop_price_jpy, equity_now_jpy, maintenance_ratio
            FROM regime_leverage_log
            ORDER BY id DESC
            LIMIT 1
            """).fetchone()
    if row is None:
        return None
    return RegimeLeverageSnapshot(
        id=row[0],
        executed_at=row[1],
        action=row[2],
        reason=row[3],
        spy_price_usd=row[4],
        usdjpy_rate=row[5],
        shares=row[6],
        entry_date=row[7],
        entry_price_jpy=row[8],
        entry_commission_jpy=row[9],
        equity_at_entry_jpy=row[10],
        stop_price_jpy=row[11],
        equity_now_jpy=row[12],
        maintenance_ratio=row[13],
    )


def insert_snapshot(decision: RegimeLeverageDecision) -> None:
    """新しい状態行を追記する。"""
    with _db_connection() as con:
        con.execute(
            """
            INSERT INTO regime_leverage_log (
                action, reason, spy_price_usd, usdjpy_rate, shares,
                entry_date, entry_price_jpy, entry_commission_jpy, equity_at_entry_jpy,
                stop_price_jpy, equity_now_jpy, maintenance_ratio
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                decision.action,
                decision.reason,
                decision.spy_price_usd,
                decision.usdjpy_rate,
                decision.shares,
                decision.entry_date,
                decision.entry_price_jpy,
                decision.entry_commission_jpy,
                decision.equity_at_entry_jpy,
                decision.stop_price_jpy,
                decision.equity_now_jpy,
                decision.maintenance_ratio,
            ],
        )
    logger.info(
        "regime_leverage_log 追記: action=%s reason=%s shares=%.4f equity_now_jpy=%.2f",
        decision.action,
        decision.reason,
        decision.shares,
        decision.equity_now_jpy,
    )
