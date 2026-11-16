"""約定・発注の結果を Discord の明細の行に写す（#1876）

明細が作れないときは None を返し、知らせは件数だけの形で出す。
"""

from typing import Any, Mapping, Optional, Sequence

from src.reporting.discord.trade_details import TradeLine
from src.utils.logger import get_logger

logger = get_logger(__name__)


def settled_trade_lines(settled: Sequence[Mapping[str, Any]]) -> list[TradeLine]:
    """PaperBroker.settle_pending_orders の結果を、約定の単価の明細にする"""
    return [
        TradeLine(
            symbol=str(row["symbol"]),
            side=int(row["side"]),
            qty=int(row["qty"]),
            price=float(row["fill_price"]),
        )
        for row in settled
    ]


def safe_settled_trade_lines(
    settled: Optional[Sequence[Mapping[str, Any]]],
) -> Optional[list[TradeLine]]:
    """約定の明細。作れなければ None"""
    if not settled:
        return None
    try:
        return settled_trade_lines(settled)
    except Exception as e:
        logger.error("約定の明細を作れない。件数だけで知らせる: %s", e, exc_info=True)
        return None


def order_trade_lines(stats: Mapping[str, Any]) -> Optional[list[TradeLine]]:
    """run_daily_orders の集計の order_details を、発注の明細にする。作れなければ None"""
    try:
        details = stats.get("order_details") or []
        lines = [
            TradeLine(
                symbol=str(d["symbol"]),
                side=int(d["side"]),
                qty=int(d["qty"]),
                price=float(d["price"]),
                is_market=bool(d["is_market"]),
            )
            for d in details
        ]
    except Exception as e:
        logger.error("発注の明細を作れない。件数だけで知らせる: %s", e, exc_info=True)
        return None
    return lines or None
