"""ペーパーの約定を保有と残高に書く（#1088 で paper_broker から移した）

呼び手は注文1件分をトランザクションで囲み、途中で落ちたら全部を戻す。
"""

from typing import Any

from src.trading.brokers.base import OrderSide
from src.utils.logger import get_logger

logger = get_logger(__name__)


def apply_fill(
    con: Any, order_id: str, symbol: str, side: int, qty: int, fill_price: float
) -> None:
    """約定に合わせてポジションと残高を更新する"""
    existing = con.execute(
        "SELECT qty, avg_price FROM paper_positions WHERE symbol=%s", [symbol]
    ).fetchone()

    if side == int(OrderSide.BUY):
        cost = qty * fill_price
        # 残高を減少
        con.execute("UPDATE paper_balance SET balance = balance - %s", [cost])
        if existing:
            old_qty, old_avg = existing
            new_qty = old_qty + qty
            new_avg = (old_avg * old_qty + fill_price * qty) / new_qty
            con.execute(
                "UPDATE paper_positions SET qty=%s, avg_price=%s, "
                "updated_at=CURRENT_TIMESTAMP WHERE symbol=%s",
                [new_qty, new_avg, symbol],
            )
        else:
            con.execute(
                "INSERT INTO paper_positions "
                "(symbol, qty, avg_price, updated_at) VALUES (%s, %s, %s, CURRENT_TIMESTAMP)",
                [symbol, qty, fill_price],
            )
    elif side == int(OrderSide.SELL):
        # 未保有銘柄の SELL は現金を計上しない（幻の現金水増しを防止）。
        # 保有数量を超える SELL は保有分にクランプする（過大な現金計上を防止）。
        if not existing:
            logger.warning(
                "[paper] %s: 未保有のため SELL をスキップ（現金・実現損益を計上しない）", symbol
            )
            return
        old_qty, old_avg = existing
        sell_qty = min(qty, old_qty)
        if sell_qty < qty:
            logger.warning(
                "[paper] %s: 売却数量 %d が保有 %d を超過 → 保有分にクランプ",
                symbol,
                qty,
                old_qty,
            )
        proceeds = sell_qty * fill_price
        con.execute("UPDATE paper_balance SET balance = balance + %s", [proceeds])
        new_qty = old_qty - sell_qty
        if new_qty <= 0:
            con.execute("DELETE FROM paper_positions WHERE symbol=%s", [symbol])
        else:
            con.execute(
                "UPDATE paper_positions SET qty=%s, "
                "updated_at=CURRENT_TIMESTAMP WHERE symbol=%s",
                [new_qty, symbol],
            )
        realized_pnl = (fill_price - old_avg) * sell_qty
        con.execute(
            "UPDATE paper_orders SET realized_pnl=%s WHERE order_id=%s",
            [realized_pnl, order_id],
        )
    elif side == int(OrderSide.SHORT):
        # 空売り新規: send_order で仮登録済みの avg_short_price を実際の約定値段で直す
        existing_short = con.execute(
            "SELECT qty, avg_short_price FROM paper_short_positions WHERE symbol=%s",
            [symbol],
        ).fetchone()
        if existing_short:
            old_qty, old_avg = existing_short
            # 仮登録の平均は、前の保有と今回の注文の price（成行は 0.0）を混ぜている。
            # その price の分を約定の値段に置き換え、前の保有と約定の加重平均にする（#1115）
            # 権利落ちの日は保有と qty が分割で直っているが、注文の行はまだ分割前なので、
            # 仮登録の分は注文の行の price と株数の積（分割で変わらない金額）で引く
            order = con.execute(
                "SELECT price, qty FROM paper_orders WHERE order_id=%s", [order_id]
            ).fetchone()
            if order and order[0] is not None and order[1] is not None:
                provisional_amount = float(order[0]) * int(order[1])
            else:
                provisional_amount = fill_price * qty
            new_avg = old_avg + (fill_price * qty - provisional_amount) / old_qty
            con.execute(
                "UPDATE paper_short_positions SET avg_short_price=%s, "
                "updated_at=CURRENT_TIMESTAMP WHERE symbol=%s",
                [new_avg, symbol],
            )
        else:
            con.execute(
                "INSERT INTO paper_short_positions "
                "(symbol, qty, avg_short_price, opened_at, updated_at) "
                "VALUES (%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                [symbol, qty, fill_price],
            )
    elif side == int(OrderSide.SHORT_COVER):
        # 空売り返済: paper_short_positions から qty を減算し実現損益を記録
        existing_short = con.execute(
            "SELECT qty, avg_short_price FROM paper_short_positions WHERE symbol=%s",
            [symbol],
        ).fetchone()
        if existing_short:
            old_qty, old_avg = existing_short
            new_qty = old_qty - qty
            if new_qty <= 0:
                con.execute("DELETE FROM paper_short_positions WHERE symbol=%s", [symbol])
            else:
                con.execute(
                    "UPDATE paper_short_positions SET qty=%s, "
                    "updated_at=CURRENT_TIMESTAMP WHERE symbol=%s",
                    [new_qty, symbol],
                )
        else:
            old_avg = fill_price  # ポジション不明時はPnL=0扱い
        realized_pnl = (old_avg - fill_price) * qty
        # 建てるときは残高を動かさないので、返済のときに損益だけを残高へ移す（#1147）
        con.execute("UPDATE paper_balance SET balance = balance + %s", [realized_pnl])
        con.execute(
            "UPDATE paper_orders SET realized_pnl=%s WHERE order_id=%s",
            [realized_pnl, order_id],
        )
