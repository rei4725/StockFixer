"""PaperBroker の空売りの仮登録と、その取り消し（paper_short_positions の書き込み）

仮登録（send_order）と取り消し（cancel_order・指値の失効）を並べ、片方だけ直して株数がずれないようにする。
呼び手が開いた接続 con を受け取り、SQL を流すだけ（#1867）。
"""

from typing import Any

from src.trading.brokers.paper import split_adjust as sa


def register_short(con: Any, sym: str, order_id: str, qty: int, price: float, splits: Any) -> None:
    """paper_short_positions に空売りを即時仮登録する（加重平均単価更新）"""
    existing_short = con.execute(
        "SELECT qty, avg_short_price, updated_at FROM paper_short_positions WHERE symbol=%s",
        [sym],
    ).fetchone()
    if existing_short:
        old_qty, old_avg, old_updated_at = existing_short
        # 権利落ちの日の寄りの前に直した行は updated_at を戻さず、分割前の注文の株数は
        # 分割を当てて足す（金額は分割で変わらないのでそのまま、#1339）
        created_at = con.execute(
            "SELECT created_at FROM paper_orders WHERE order_id=%s", [order_id]
        ).fetchone()[0]
        factor = sa.split_factor_between(splits, created_at, old_updated_at)
        new_qty = old_qty + int(round(qty * factor))
        new_avg = (old_avg * old_qty + price * qty) / new_qty
        con.execute(
            "UPDATE paper_short_positions SET qty=%s, avg_short_price=%s, "
            "updated_at=GREATEST(updated_at, CURRENT_TIMESTAMP) WHERE symbol=%s",
            [new_qty, new_avg, sym],
        )
    else:
        con.execute(
            "INSERT INTO paper_short_positions "
            "(symbol, qty, avg_short_price, opened_at, updated_at) "
            "VALUES (%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            [sym, qty, price],
        )


def undo_short_registration(
    con: Any, symbol: str, qty: int, price: Any, splits: Any = None, created_at: Any = None
) -> None:
    """取り消した空売りの分を paper_short_positions から戻す（加重平均単価も戻す, #1104）

    splits と created_at を渡すと、仮登録で分割を当てて足した株数を引く（#1339）。
    """
    row = con.execute(
        "SELECT qty, avg_short_price, updated_at FROM paper_short_positions WHERE symbol=%s",
        [symbol],
    ).fetchone()
    if not row:
        return
    cur_qty, cur_avg = int(row[0]), float(row[1] or 0.0)
    factor = 1.0 if created_at is None else sa.split_factor_between(splits, created_at, row[2])
    rest = cur_qty - int(round(qty * factor))
    if rest <= 0:
        con.execute("DELETE FROM paper_short_positions WHERE symbol=%s", [symbol])
        return
    rest_avg = (cur_avg * cur_qty - float(price or 0.0) * qty) / rest
    # 権利落ちの日の寄りの前に直した行の updated_at を、前の日と読まれる今の時刻に戻さない（#1339）
    con.execute(
        "UPDATE paper_short_positions SET qty=%s, avg_short_price=%s, "
        "updated_at=GREATEST(updated_at, CURRENT_TIMESTAMP) WHERE symbol=%s",
        [rest, rest_avg, symbol],
    )
