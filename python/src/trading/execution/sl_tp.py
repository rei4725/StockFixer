"""SL/TP トリガーによる保有ポジションの強制クローズ。"""

import math

from src.domain.ports import MarketDataPort, TradeDiffSink
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.risk_manager import RiskManager
from src.trading.types import TradingGateStatus
from src.utils.logger import get_logger

from .pending import _load_pending_exits, _log_pending_skips
from .recording import _record_order, _recorder
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
    """保有ポジションを走査し SL/TP トリガーで成行の決済を発行する。

    買い持ちは成行売り、空売りは SHORT_COVER の成行で返済する（#1120）。

    Returns:
        SL/TP が発動して決済の注文を出した銘柄コードのセット。
        呼び出し元は prediction ベースの売りシグナルから除外する。
    """
    triggered: set[str] = set()
    record = _recorder(_record_order, stats)  # 記録の失敗でも売りは出たものとして除外する (#1104)
    try:
        short_positions = broker.get_short_positions()
    except AttributeError:
        short_positions = []
    exits = [(pos, False) for pos in broker.get_positions()]
    exits += [(pos, True) for pos in short_positions]
    pending = _load_pending_exits(broker)
    pending_skipped: list[str] = []
    for pos, is_short in exits:
        symbol = str(pos.get("symbol", "")).replace(".T", "")
        qty = int(pos.get("qty") or 0)
        avg_price = float(pos.get("avg_short_price" if is_short else "avg_price") or 0.0)
        current_price = float(pos.get("current_price") or 0.0)
        if qty > 0 and avg_price > 0 and math.isnan(current_price):
            # 現在値が取れない保有を損益0として素通りさせない（#844）
            logger.warning("[exec] %s: 現在値が取れず SL/TP を評価できない（%d株）", symbol, qty)
            continue
        if qty <= 0 or avg_price <= 0 or current_price <= 0:
            continue

        risk = RiskManager(broker, market, symbol)
        if is_short:
            sl_triggered, tp_triggered, reason = risk.check_sl_tp(
                avg_price, current_price, is_short=True
            )
        else:
            sl_triggered, tp_triggered, reason = risk.check_sl_tp(avg_price, current_price)
        if not sl_triggered and not tp_triggered:
            continue

        side = OrderSide.SHORT_COVER if is_short else OrderSide.SELL
        if pending.has(symbol, side):
            pending_skipped.append(symbol)
            continue
        try:
            # 損失の止めは約定を優先する。流動性の判定による指値は窓を開けた日に失効するため使わない（#881）
            order_type, order_price, order_session = OrderType.MARKET, 0.0, "open"
            result = broker.send_order(
                symbol,
                side,
                qty,
                price=order_price,
                order_type=order_type,
            )
            record(
                market=market,
                predicted_at="",
                symbol=symbol,
                side=side,
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
                "[exec] SL/TP発動 → %s: %s %d株 @ %.1f (%s)",
                "成行の空売り返済" if is_short else "成行売り",
                symbol,
                qty,
                current_price,
                reason,
            )
            if is_short:
                stats["short_orders"] += 1
            else:
                stats["sell_orders"] += 1
            stats["total_turnover"] += current_price * qty
            triggered.add(symbol)
        except Exception:
            logger.error("[exec] SL/TP売り注文エラー (%s)", symbol, exc_info=True)
            stats["errors"] += 1

    _log_pending_skips("SL/TP", pending_skipped)
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
