"""paper_real_diff テーブル: paper / real 約定価格の乖離追跡のアダプタ。"""

from datetime import datetime
from typing import Any

from src.domain.ports import TradeDiffSink
from src.domain.types import TradeDiffRecord
from src.utils.db import db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PostgresTradeDiffSink(TradeDiffSink):
    """paper_real_diff テーブルへ書き込む TradeDiffSink 実装。

    同一キー (market, symbol, predicted_at, side) の既存行があれば
    その値を引き継いだうえで DELETE + INSERT で置き換える。
    """

    def record(self, record: TradeDiffRecord) -> None:
        checked_at = record.checked_at or datetime.now()
        with db_connection() as con:
            row = con.execute(
                """
            SELECT signal_price, paper_order_id, real_order_id, paper_price, real_price,
                   paper_slippage, real_slippage, paper_filled_at, real_checked_at, created_at,
                   order_session, split_ratio
            FROM paper_real_diff
            WHERE market = %s AND symbol = %s AND predicted_at = %s AND side = %s
            """,
                [record.market, record.symbol, record.predicted_at, record.side],
            ).fetchone()

            merged: dict[str, Any] = {
                "signal_price": record.signal_price,
                "paper_order_id": None,
                "real_order_id": None,
                "paper_price": None,
                "real_price": None,
                "paper_slippage": None,
                "real_slippage": None,
                "paper_filled_at": None,
                "real_checked_at": None,
                "created_at": checked_at,
                "order_session": record.order_session,
                "split_ratio": record.split_ratio,
            }
            if row:
                merged.update(
                    {
                        "signal_price": (
                            float(row[0]) if row[0] is not None else record.signal_price
                        ),
                        "paper_order_id": row[1],
                        "real_order_id": row[2],
                        "paper_price": row[3],
                        "real_price": row[4],
                        "paper_slippage": row[5],
                        "real_slippage": row[6],
                        "paper_filled_at": row[7],
                        "real_checked_at": row[8],
                        "created_at": row[9] or checked_at,
                        "order_session": row[10] or record.order_session,
                        "split_ratio": row[11] if len(row) > 11 else record.split_ratio,
                    }
                )
            if record.split_ratio is not None:
                merged["split_ratio"] = record.split_ratio

            merged["signal_price"] = record.signal_price
            if record.mode == "paper":
                merged["paper_order_id"] = record.order_id
                if record.actual_price is not None:
                    merged["paper_price"] = record.actual_price
                    merged["paper_slippage"] = (
                        (record.actual_price - record.signal_price) / record.signal_price
                        if record.signal_price
                        else None
                    )
                    merged["paper_filled_at"] = checked_at
            else:
                merged["real_order_id"] = record.order_id
                if record.actual_price is not None:
                    merged["real_price"] = record.actual_price
                    merged["real_slippage"] = (
                        (record.actual_price - record.signal_price) / record.signal_price
                        if record.signal_price
                        else None
                    )
                    merged["real_checked_at"] = checked_at

            price_diff = None
            if merged["paper_price"] is not None and merged["real_price"] is not None:
                price_diff = float(merged["real_price"]) - float(merged["paper_price"])

            con.execute(
                "DELETE FROM paper_real_diff WHERE market = %s AND symbol = %s "
                "AND predicted_at = %s AND side = %s",
                [record.market, record.symbol, record.predicted_at, record.side],
            )
            con.execute(
                """
            INSERT INTO paper_real_diff (
                market, symbol, predicted_at, side, signal_price,
                paper_order_id, real_order_id, paper_price, real_price,
                paper_slippage, real_slippage, price_diff,
                paper_filled_at, real_checked_at, created_at, updated_at, order_session,
                split_ratio
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    CURRENT_TIMESTAMP, %s, %s)
            """,
                [
                    record.market,
                    record.symbol,
                    record.predicted_at,
                    record.side,
                    merged["signal_price"],
                    merged["paper_order_id"],
                    merged["real_order_id"],
                    merged["paper_price"],
                    merged["real_price"],
                    merged["paper_slippage"],
                    merged["real_slippage"],
                    price_diff,
                    merged["paper_filled_at"],
                    merged["real_checked_at"],
                    merged["created_at"],
                    merged["order_session"],
                    merged.get("split_ratio"),
                ],
            )
