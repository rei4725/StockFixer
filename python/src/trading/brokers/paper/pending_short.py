"""約定待ちの空売り（send_order の仮登録）を、約定した空売りと分ける（#1225）

paper_short_positions は send_order の時点で約定待ちの SHORT を足している（成行は単価 0.0）。
評価と返済は約定した分だけで行い、保有の上限の数え（#1146）には仮登録も残す。
"""

from typing import Any

from src.trading.brokers.base import OrderSide
from src.trading.brokers.paper import split_adjust as sa


def read_pending_shorts(con: Any) -> dict[str, list[tuple[int, float, Any]]]:
    """約定待ちの SHORT の銘柄ごとの注文 (株数, 仮登録の金額 = price × 株数, created_at) を返す"""
    rows = con.execute(
        "SELECT symbol, qty, COALESCE(price, 0) * qty, created_at FROM paper_orders "
        "WHERE status='pending' AND side=%s",
        [int(OrderSide.SHORT)],
    ).fetchall()
    pending: dict[str, list[tuple[int, float, Any]]] = {}
    for sym, qty, amount, created_at in rows:
        pending.setdefault(str(sym), []).append((int(qty or 0), float(amount or 0.0), created_at))
    return pending


def pending_in_row_units(
    orders: list[tuple[int, float, Any]] | None, splits: Any, row_updated_at: Any
) -> tuple[int, float] | None:
    """約定待ちの注文を、保有の行の単位の (株数, 金額) にまとめる（#1339）

    権利落ちの日の寄りの前に直した行へ重ねた SHORT は、仮登録で分割を当てて足してある。
    注文の行は分割前のままなので、注文の後で行の updated_at までに権利落ちした分割の比で株数を直す。
    """
    if not orders:
        return None
    qty = sum(
        int(round(q * sa.split_factor_between(splits, created_at, row_updated_at)))
        for q, _, created_at in orders
    )
    return qty, sum(amount for _, amount, _ in orders)


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
