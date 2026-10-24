"""SL/TP トリガーによる保有ポジションの強制クローズ。"""

import math

from src.domain.ports import MarketDataPort, TradeDiffSink
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.risk_manager import RiskManager
from src.trading.types import TradingGateStatus
from src.utils.logger import get_logger

from .recording import _record_order
from .stats import OrderExecutionStats

logger = get_logger(__name__)


def _check_sl_tp_exits(
    broker: BrokerBase,
    market: str,
    mode: str,
    market_data: MarketDataPort | None,
    stats: OrderExecutionStats,
    trade_diff_sink: TradeDiffSink,
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
        if qty > 0 and avg_price > 0 and math.isnan(current_price):
            # 現在値が取れない保有を損益0として素通りさせない（#844）
            logger.warning("[exec] %s: 現在値が取れず SL/TP を評価できない（%d株）", symbol, qty)
            continue
        if qty <= 0 or avg_price <= 0 or current_price <= 0:
            continue

        risk = RiskManager(broker, market, symbol)
        sl_triggered, tp_triggered, reason = risk.check_sl_tp(avg_price, current_price)
        if not sl_triggered and not tp_triggered:
            continue

        try:
            # 損失の止めは約定を優先する。流動性の判定による指値は窓を開けた日に失効するため使わない（#881）
            order_type, order_price, order_session = OrderType.MARKET, 0.0, "open"
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
                trade_diff_sink=trade_diff_sink,
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


def _sl_tp_exits_only(
    broker: BrokerBase,
    market: str,
    mode: str,
    market_data: MarketDataPort | None,
    stats: OrderExecutionStats,
    trade_diff_sink: TradeDiffSink,
    gate_status: TradingGateStatus,
) -> None:
    """予測を使えない日に、当日の損失を stats に載せて SL/TP の決済だけ行う（#1049・#932）。"""
    stats.update(
        {
            "daily_loss": gate_status.daily_loss,
            "daily_loss_limit": gate_status.daily_loss_limit,
        }
    )
    _check_sl_tp_exits(broker, market, mode, market_data, stats, trade_diff_sink)
