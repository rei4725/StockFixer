"""約定待ちの決済の確かめ — 同じ銘柄に2本目の決済を出さない（#1117）。

get_positions は約定前の SELL を引かないため、前の回（16:30 のルールの道など）の
約定待ちを見ずに売ると二重の決済になる。ホライズン決済も同じ読み取りを使う（#1168）。
"""

from dataclasses import dataclass, field

from src.trading.brokers.base import BrokerBase, OrderSide
from src.utils.db import db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PendingExits:
    """約定待ちの SELL と SHORT_COVER がある銘柄コード。"""

    sells: frozenset[str] = field(default_factory=frozenset)
    covers: frozenset[str] = field(default_factory=frozenset)

    def has(self, symbol: str, side: OrderSide) -> bool:
        symbols = self.covers if side == OrderSide.SHORT_COVER else self.sells
        return symbol.replace(".T", "") in symbols


def _load_pending_exits(broker: BrokerBase) -> PendingExits:
    """paper_orders から約定待ちの決済の銘柄を読む。paper 以外のブローカーは空を返す。"""
    if broker.broker_name != "paper":
        return PendingExits()
    return _read_pending_exits()


def _read_pending_exits() -> PendingExits:
    """paper_orders から約定待ちの決済の銘柄を読む（紙の保有だけを扱う道が直接呼ぶ。#1168）。"""
    try:
        with db_connection() as con:
            rows = con.execute(
                "SELECT DISTINCT symbol, side FROM paper_orders "
                "WHERE status = 'pending' AND side IN (%s, %s)",
                [int(OrderSide.SELL), int(OrderSide.SHORT_COVER)],
            ).fetchall()
    except Exception:
        logger.error("[exec] 約定待ちの決済を読めない（重ねの確かめなしで続ける）", exc_info=True)
        return PendingExits()
    sells = frozenset(str(s) for s, side in rows if int(side) == int(OrderSide.SELL))
    covers = frozenset(str(s) for s, side in rows if int(side) == int(OrderSide.SHORT_COVER))
    return PendingExits(sells=sells, covers=covers)


def _log_pending_skips(label: str, skipped: list[str]) -> None:
    """約定待ちのため決済を飛ばした件数を出す。"""
    if skipped:
        logger.info(
            "[exec] %s: 約定待ちの決済があるため %d 件をスキップ: %s",
            label,
            len(skipped),
            skipped,
        )
