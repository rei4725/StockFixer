"""注文結果の記録と live 約定差分の同期。"""

import uuid
from datetime import date, timedelta
from typing import Any

from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.utils.db import upsert_paper_real_diff
from src.utils.db._connection import _db_connection


def _link_paper_order_metadata(
    order_id: str,
    market: str,
    predicted_at: str,
    signal_price: float,
    horizon: int | None = None,
    target_exit_date: str | None = None,
) -> None:
    with _db_connection() as con:
        con.execute(
            """
            UPDATE paper_orders
            SET market = ?, predicted_at = ?, signal_price = ?,
                horizon = ?, target_exit_date = ?
            WHERE order_id = ?
            """,
            [market, predicted_at, signal_price, horizon, target_exit_date, order_id],
        )


def _sync_live_execution_diffs(broker: BrokerBase) -> None:
    for order in broker.get_orders():
        order_id = str(order.get("order_id") or "")
        price = order.get("price")
        if not order_id or price in (None, ""):
            continue
        try:
            actual_price = float(price)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if actual_price <= 0:
            continue

        with _db_connection() as con:
            row = con.execute(
                """
                SELECT market, symbol, predicted_at, side, signal_price
                FROM paper_real_diff
                WHERE real_order_id = ?
                """,
                [order_id],
            ).fetchone()
        if row is None:
            continue

        upsert_paper_real_diff(
            market=str(row[0]),
            symbol=str(row[1]),
            predicted_at=str(row[2]),
            side=int(row[3]),
            signal_price=float(row[4] or 0.0),
            mode="live",
            order_id=order_id,
            actual_price=actual_price,
        )


def _record_order(
    market: str,
    predicted_at: str,
    symbol: str,
    side: OrderSide,
    qty: int,
    signal_price: float,
    order_price: float,
    order_type: OrderType,
    order_result: dict[str, Any],
    broker: BrokerBase,
    mode: str,
    order_session: str = "open",
    split_ratio: float = 1.0,
    horizon: int | None = None,
) -> None:
    """注文結果を orders テーブルに保存する"""
    with _db_connection() as con:
        con.execute(
            """
            INSERT INTO orders
                (order_id, symbol, side, qty, price, order_type, status, broker, mode, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                order_result.get("order_id", str(uuid.uuid4())[:12]),
                symbol,
                int(side),
                qty,
                order_price,
                int(order_type),
                order_result.get("status", "unknown"),
                broker.broker_name,
                mode,
            ],
        )

    order_id = str(order_result.get("order_id", ""))
    if mode == "paper" and order_id:
        target_exit_date: str | None = None
        if horizon is not None and side in (OrderSide.BUY, OrderSide.SHORT):
            target_exit_date = (date.today() + timedelta(days=horizon)).isoformat()
        _link_paper_order_metadata(
            order_id, market, predicted_at, signal_price, horizon, target_exit_date
        )

    fill_price_raw = order_result.get("fill_price")
    fill_price = float(fill_price_raw) if isinstance(fill_price_raw, (int, float)) else None

    upsert_paper_real_diff(
        market=market,
        symbol=symbol,
        predicted_at=predicted_at,
        side=int(side),
        signal_price=signal_price,
        mode=mode,
        order_id=order_id,
        actual_price=fill_price,
        order_session=order_session,
        split_ratio=split_ratio,
    )
