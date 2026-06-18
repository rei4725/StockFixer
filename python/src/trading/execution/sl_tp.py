"""SL/TP トリガーによる保有ポジションの強制クローズ。"""

from src.domain.ports import MarketDataPort
from src.trading.brokers.base import BrokerBase, OrderSide
from src.trading.risk_manager import RiskManager
from src.utils.logger import get_logger

from .params import _choose_order_params
from .recording import _record_order
from .stats import OrderExecutionStats

logger = get_logger(__name__)


def _check_sl_tp_exits(
    broker: BrokerBase,
    market: str,
    mode: str,
    market_data: MarketDataPort | None,
    stats: OrderExecutionStats,
) -> set[str]:
    """保有ポジションを走査し SL/TP トリガーで成行売りを発行する。

    Returns:
        SL/TP が発動して売り注文を出した銘柄コードのセット。
        呼び出し元は prediction ベースの売りシグナルから除外する。
    """
    triggered: set[str] = set()
    positions = broker.get_positions()
    for pos in positions:
        symbol = str(pos.get("symbol", "")).replace(".T", "")
        qty = int(pos.get("qty") or 0)
        avg_price = float(pos.get("avg_price") or 0.0)
        current_price = float(pos.get("current_price") or 0.0)
        if qty <= 0 or avg_price <= 0 or current_price <= 0:
            continue

        risk = RiskManager(broker, market, symbol)
        sl_triggered, tp_triggered, reason = risk.check_sl_tp(avg_price, current_price)
        if not sl_triggered and not tp_triggered:
            continue

        try:
            order_type, order_price, order_reason, order_session = _choose_order_params(
                market=market,
                symbol=symbol,
                side=OrderSide.SELL,
                current_price=current_price,
                market_data=market_data,
            )
            result = broker.send_order(
                symbol,
                OrderSide.SELL,
                qty,
                price=order_price,
                order_type=order_type,
            )
            _record_order(
                market=market,
                predicted_at="",
                symbol=symbol,
                side=OrderSide.SELL,
                qty=qty,
                signal_price=current_price,
                order_price=order_price,
                order_type=order_type,
                order_result=result,
                broker=broker,
                mode=mode,
                order_session=order_session,
            )
            logger.info(
                "[exec] SL/TP発動 → 成行売り: %s %d株 @ %.1f (%s)",
                symbol,
                qty,
                current_price,
                reason,
            )
            stats["sell_orders"] += 1
            stats["total_turnover"] += current_price * qty
            triggered.add(symbol)
        except Exception:
            logger.error("[exec] SL/TP売り注文エラー (%s)", symbol, exc_info=True)
            stats["errors"] += 1

    return triggered
