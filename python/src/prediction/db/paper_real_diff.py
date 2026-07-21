"""paper_real_diff テーブル: paper / real 約定価格の乖離追跡。"""

from datetime import datetime, timedelta
from typing import Any

from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def upsert_paper_real_diff(
    market: str,
    symbol: str,
    predicted_at: str,
    side: int,
    signal_price: float,
    mode: str,
    order_id: str,
    actual_price: float | None = None,
    checked_at: datetime | None = None,
    order_session: str = "open",
    split_ratio: float | None = None,
) -> None:
    """paper / real の価格差追跡テーブルを更新する。

    order_session: "open"（寄付）または "close"（引け）。
    """
    checked_at = checked_at or datetime.now()
    with _db_connection() as con:
        row = con.execute(
            """
            SELECT signal_price, paper_order_id, real_order_id, paper_price, real_price,
                   paper_slippage, real_slippage, paper_filled_at, real_checked_at, created_at,
                   order_session, split_ratio
            FROM paper_real_diff
            WHERE market = %s AND symbol = %s AND predicted_at = %s AND side = %s
            """,
            [market, symbol, predicted_at, side],
        ).fetchone()

        merged: dict[str, Any] = {
            "signal_price": signal_price,
            "paper_order_id": None,
            "real_order_id": None,
            "paper_price": None,
            "real_price": None,
            "paper_slippage": None,
            "real_slippage": None,
            "paper_filled_at": None,
            "real_checked_at": None,
            "created_at": checked_at,
            "order_session": order_session,
            "split_ratio": split_ratio,
        }
        if row:
            merged.update(
                {
                    "signal_price": (float(row[0]) if row[0] is not None else signal_price),
                    "paper_order_id": row[1],
                    "real_order_id": row[2],
                    "paper_price": row[3],
                    "real_price": row[4],
                    "paper_slippage": row[5],
                    "real_slippage": row[6],
                    "paper_filled_at": row[7],
                    "real_checked_at": row[8],
                    "created_at": row[9] or checked_at,
                    "order_session": row[10] or order_session,
                    "split_ratio": row[11] if len(row) > 11 else split_ratio,
                }
            )
        if split_ratio is not None:
            merged["split_ratio"] = split_ratio

        merged["signal_price"] = signal_price
        if mode == "paper":
            merged["paper_order_id"] = order_id
            if actual_price is not None:
                merged["paper_price"] = actual_price
                merged["paper_slippage"] = (
                    (actual_price - signal_price) / signal_price if signal_price else None
                )
                merged["paper_filled_at"] = checked_at
        else:
            merged["real_order_id"] = order_id
            if actual_price is not None:
                merged["real_price"] = actual_price
                merged["real_slippage"] = (
                    (actual_price - signal_price) / signal_price if signal_price else None
                )
                merged["real_checked_at"] = checked_at

        price_diff = None
        if merged["paper_price"] is not None and merged["real_price"] is not None:
            price_diff = float(merged["real_price"]) - float(merged["paper_price"])

        con.execute(
            "DELETE FROM paper_real_diff WHERE market = %s AND symbol = %s "
            "AND predicted_at = %s AND side = %s",
            [market, symbol, predicted_at, side],
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
                market,
                symbol,
                predicted_at,
                side,
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


def load_paper_real_diff_summary(recent_days: int = 7) -> dict:
    """直近期間の paper / real 乖離サマリーを返す。"""
    since = datetime.now() - timedelta(days=recent_days)
    with _db_connection() as con:
        row = con.execute(
            """
            SELECT
                COUNT(*) AS tracked_count,
                COUNT(*) FILTER (
                    WHERE paper_price IS NOT NULL AND real_price IS NOT NULL
                ) AS comparable_count,
                AVG(paper_slippage) AS avg_paper_slippage,
                AVG(real_slippage) AS avg_real_slippage,
                AVG(ABS(price_diff)) AS avg_abs_price_diff,
                AVG(ABS(price_diff / NULLIF(signal_price, 0))) AS avg_abs_diff_ratio,
                MAX(ABS(price_diff)) AS max_abs_price_diff
            FROM paper_real_diff
            WHERE COALESCE(paper_filled_at, real_checked_at, created_at) >= %s
            """,
            [since],
        ).fetchone()

    return {
        "tracked_count": int(row[0] or 0),
        "comparable_count": int(row[1] or 0),
        "avg_paper_slippage": float(row[2] or 0.0),
        "avg_real_slippage": float(row[3] or 0.0),
        "avg_abs_price_diff": float(row[4] or 0.0),
        "avg_abs_diff_ratio": float(row[5] or 0.0),
        "max_abs_price_diff": float(row[6] or 0.0),
    }


def load_open_close_advantage_summary(recent_days: int = 30) -> dict[str, dict]:
    """寄付 vs 引けの価格優位性をセッション別に集計する（R-405）。

    Returns:
        {"open": {...}, "close": {...}} — セッションがない場合は空辞書。
        各値: count, avg_slippage, avg_abs_slippage, min_slippage, max_slippage
    """
    since = datetime.now() - timedelta(days=recent_days)
    with _db_connection() as con:
        rows = con.execute(
            """
            SELECT
                COALESCE(order_session, 'open') AS session,
                COUNT(*) AS cnt,
                AVG(paper_slippage) AS avg_slippage,
                AVG(ABS(paper_slippage)) AS avg_abs_slippage,
                MIN(paper_slippage) AS min_slippage,
                MAX(paper_slippage) AS max_slippage
            FROM paper_real_diff
            WHERE paper_price IS NOT NULL
              AND COALESCE(paper_filled_at, created_at) >= %s
            GROUP BY session
            ORDER BY session
            """,
            [since],
        ).fetchall()

    result: dict[str, dict] = {}
    for row in rows:
        session = str(row[0])
        result[session] = {
            "count": int(row[1] or 0),
            "avg_slippage": float(row[2] or 0.0),
            "avg_abs_slippage": float(row[3] or 0.0),
            "min_slippage": float(row[4] or 0.0),
            "max_slippage": float(row[5] or 0.0),
        }
    return result
