"""約定待ちの空売り（send_order の仮登録）を、約定した空売りと分ける（#1225）

paper_short_positions は send_order の時点で約定待ちの SHORT を足している（成行は単価 0.0）。
評価と返済は約定した分だけで行い、保有の上限の数え（#1146）には仮登録も残す。
"""

from typing import Any

from src.trading.brokers.base import OrderSide


def read_pending_shorts(con: Any) -> dict[str, tuple[int, float]]:
    """約定待ちの SHORT の銘柄ごとの (株数, 仮登録の金額 = 注文の price × 株数) を返す"""
    rows = con.execute(
        "SELECT symbol, SUM(qty), SUM(COALESCE(price, 0) * qty) FROM paper_orders "
        "WHERE status='pending' AND side=%s GROUP BY symbol",
        [int(OrderSide.SHORT)],
    ).fetchall()
    return {str(sym): (int(qty or 0), float(amount or 0.0)) for sym, qty, amount in rows}


def split_filled_short(
    pending: tuple[int, float] | None, raw_qty: int, qty: int, avg: float
) -> tuple[int, int, float]:
    """空売りの保有を (約定待ちの株数, 約定した株数, 約定した分の平均単価) に分ける

    qty・avg は分割で直した後の値。注文の行は分割前なので、株数は保有の直しの比で直し、
    金額（分割で変わらない）はそのまま引く。約定した分が無ければ平均単価は avg のまま返す。
    """
    if not pending or pending[0] <= 0:
        return 0, qty, avg
    pending_raw_qty, pending_amount = pending
    ratio = qty / raw_qty if raw_qty else 1.0
    pending_qty = min(qty, int(round(pending_raw_qty * ratio)))
    filled_qty = qty - pending_qty
    if filled_qty <= 0:
        return qty, 0, avg
    return pending_qty, filled_qty, (avg * qty - pending_amount) / filled_qty
