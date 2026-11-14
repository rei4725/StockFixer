"""新規の建て（予想の買い・空売りの新規）の候補を回して発注する手順（runner.py から分けた）。

発注の値の決め方（choose_params）と保有の上限（max_positions）は呼び出し側から受け取る。
"""

from collections import Counter
from typing import Callable

import pandas as pd

from config.settings import MIN_CHANGE_RATIO
from src.domain.ports import MarketDataPort, TradeDiffSink
from src.trading.brokers.base import BrokerBase, OrderSide
from src.trading.position_size_reason import zero_qty_skip_key
from src.trading.risk_manager import RiskManager
from src.utils.logger import get_logger

from .params import (
    _SPLIT_LOW_CONFIDENCE,
    _apply_split_qty,
    _calc_split_ratio,
    _resolve_kelly_params,
)
from .selection import _determine_entry_horizon, _orders_cap_reached, _sector_has_room
from .stats import OrderExecutionStats, count_skip

# runner.py から分けた手順なので、ログの名前は runner のままにする（ログを見る側を変えない）
logger = get_logger("src.trading.execution.runner")


def place_new_buys(
    buy_signals: pd.DataFrame,
    *,
    broker: BrokerBase,
    risk: RiskManager,
    lot: int,
    stats: OrderExecutionStats,
    record: Callable[..., None],
    mode: str,
    market_data: MarketDataPort | None,
    trade_diff_sink: TradeDiffSink,
    position_count: int,
    max_positions: int,
    choose_params: Callable[..., tuple],
    sector_counts: Counter[str],
) -> int:
    """新規買いの候補を並びのまま回して発注し、発注後の建玉数を返す。

    1回の発注の上限と業種の枠は、発注した銘柄だけに数える（#1759）。sector_counts は
    保有と約定待ちの業種の数から始め、発注するたびに足す。
    """
    sent = 0
    for _, row in buy_signals.iterrows():
        symbol = row["symbol"]
        current_price = float(row.get("current_price") or 0)

        if position_count >= max_positions:
            logger.info("[exec] 保有上限 %d 到達のため以降の新規買いをスキップ", max_positions)
            count_skip(stats, "max_positions")
            continue

        if _orders_cap_reached(sent):
            logger.info("[exec] %s: 1回の発注の上限に届いたため新規買いをスキップ", symbol)
            count_skip(stats, "max_orders_per_run")
            continue

        if abs(float(row.get("diff_ratio") or 0.0)) < MIN_CHANGE_RATIO:  # 変動が小さい（R-214）
            logger.info(
                "[exec] %s: diff_ratio=%.3f%% < min_change=%.3f%% → スキップ",
                symbol,
                float(row.get("diff_ratio") or 0) * 100,
                MIN_CHANGE_RATIO * 100,
            )
            count_skip(stats, "min_change")
            continue

        if current_price <= 0:
            logger.warning(f"[exec] {symbol}: 現在値が取得できないためスキップ")
            count_skip(stats, "no_price")
            continue

        confidence = float(row.get("confidence_ratio") or 1.0)
        split_ratio = _calc_split_ratio(confidence)
        if split_ratio == 0.0:
            logger.info(
                "[exec] %s: confidence_ratio=%.3f < %.2f → 見送り (R-308)",
                symbol,
                confidence,
                _SPLIT_LOW_CONFIDENCE,
            )
            count_skip(stats, "low_confidence")
            continue

        try:
            kelly_win_rate, kelly_avg_win, kelly_avg_loss = _resolve_kelly_params(
                str(row["market"]), symbol
            )
            qty = risk.calc_position_size(
                symbol,
                current_price,
                win_rate=kelly_win_rate,
                avg_win=kelly_avg_win,
                avg_loss=kelly_avg_loss,
                confidence_ratio=confidence,
            )
            if qty <= 0:
                logger.info(f"[exec] {symbol}: 発注株数 0 → スキップ（{risk.zero_reason}）")
                count_skip(stats, zero_qty_skip_key(risk.zero_reason))
                continue

            if split_ratio < 1.0:
                qty = _apply_split_qty(qty, split_ratio, lot=lot)
                logger.info(
                    "[exec] %s: confidence_ratio=%.3f → %.0f%%発注 %d株 (R-308)",
                    symbol,
                    confidence,
                    split_ratio * 100,
                    qty,
                )

            sector = str(row.get("sector", ""))
            if not _sector_has_room(sector, sector_counts):
                logger.info("[exec] %s: 同一業種の上限で新規買いをスキップ", symbol)
                count_skip(stats, "sector_cap")
                continue

            order_type, order_price, order_reason, order_session = choose_params(
                market=str(row["market"]),
                symbol=symbol,
                side=OrderSide.BUY,
                current_price=current_price,
                market_data=market_data,
            )
            result = broker.send_order(
                symbol,
                OrderSide.BUY,
                qty,
                price=order_price,
                order_type=order_type,
            )
            record(
                market=str(row["market"]),
                predicted_at=str(row["predicted_at"]),
                symbol=symbol,
                side=OrderSide.BUY,
                qty=qty,
                signal_price=current_price,
                order_price=order_price,
                order_type=order_type,
                order_result=result,
                broker=broker,
                mode=mode,
                order_session=order_session,
                split_ratio=split_ratio,
                horizon=_determine_entry_horizon(row),
                trade_diff_sink=trade_diff_sink,
            )
            logger.info(
                f"[exec] 買い発注: {symbol} {qty}株 @ {order_type.name}({order_reason}) "
                f"(1d変化率={row['diff_ratio']:.3%}, 統合スコア={row['multi_horizon_score']:.3%}, "
                f"閾値={row['effective_buy_threshold']:.3%}, sector={row.get('sector', 'N/A')}, "
                f"split={split_ratio:.0%})"
            )
            stats["buy_orders"] += 1
            stats["total_turnover"] += current_price * qty
            position_count += 1
            sent += 1
            sector_counts[sector] += 1
        except Exception as e:
            logger.error(f"[exec] 買い注文エラー ({symbol}): {e}", exc_info=True)
            stats["errors"] += 1
    return position_count


def place_new_shorts(
    short_entry_candidates: pd.DataFrame,
    *,
    broker: BrokerBase,
    risk: RiskManager,
    lot: int,
    stats: OrderExecutionStats,
    record: Callable[..., None],
    mode: str,
    market_data: MarketDataPort | None,
    trade_diff_sink: TradeDiffSink,
    position_count: int,
    max_positions: int,
    choose_params: Callable[..., tuple],
) -> int:
    """空売りの新規の候補を並びのまま回して発注し、発注後の建玉数を返す。

    1回の発注の上限は、建てた数で数える（#1759）。
    """
    sent = 0
    for _, row in short_entry_candidates.iterrows():
        symbol = row["symbol"]
        current_price = float(row.get("current_price") or 0)

        if position_count >= max_positions:
            logger.info("[exec] 保有上限 %d 到達で新規ショートをスキップ", max_positions)
            count_skip(stats, "max_positions")
            continue

        if _orders_cap_reached(sent):
            logger.info("[exec] %s: 1回の発注の上限に届いたため新規ショートをスキップ", symbol)
            count_skip(stats, "max_orders_per_run")
            continue

        if abs(float(row.get("diff_ratio") or 0.0)) < MIN_CHANGE_RATIO:
            logger.info(
                "[exec] %s: diff_ratio=%.3f%% < min_change=%.3f%% → ショートスキップ",
                symbol,
                float(row.get("diff_ratio") or 0) * 100,
                MIN_CHANGE_RATIO * 100,
            )
            count_skip(stats, "min_change")
            continue

        if current_price <= 0:
            logger.warning(f"[exec] {symbol}: 現在値が取得できないためショートスキップ")
            count_skip(stats, "no_price")
            continue

        confidence = float(row.get("confidence_ratio") or 1.0)
        short_split_ratio = _calc_split_ratio(confidence)
        if short_split_ratio == 0.0:
            logger.info(
                "[exec] %s: confidence_ratio=%.3f < %.2f → ショート見送り (R-308)",
                symbol,
                confidence,
                _SPLIT_LOW_CONFIDENCE,
            )
            count_skip(stats, "low_confidence")
            continue

        try:
            kelly_win_rate, kelly_avg_win, kelly_avg_loss = _resolve_kelly_params(
                str(row["market"]), symbol
            )
            qty = risk.calc_position_size(
                symbol,
                current_price,
                win_rate=kelly_win_rate,
                avg_win=kelly_avg_win,
                avg_loss=kelly_avg_loss,
                confidence_ratio=confidence,
            )
            if qty <= 0:
                logger.info(f"[exec] {symbol}: 発注株数 0 → ショートせず（{risk.zero_reason}）")
                count_skip(stats, zero_qty_skip_key(risk.zero_reason))
                continue

            if short_split_ratio < 1.0:
                qty = _apply_split_qty(qty, short_split_ratio, lot=lot)
                logger.info(
                    "[exec] %s: confidence_ratio=%.3f → %.0f%%ショート発注 %d株 (R-308)",
                    symbol,
                    confidence,
                    short_split_ratio * 100,
                    qty,
                )

            order_type, order_price, order_reason, order_session = choose_params(
                market=str(row["market"]),
                symbol=symbol,
                side=OrderSide.SHORT,
                current_price=current_price,
                market_data=market_data,
            )
            result = broker.send_order(
                symbol,
                OrderSide.SHORT,
                qty,
                price=order_price,
                order_type=order_type,
            )
            record(
                market=str(row["market"]),
                predicted_at=str(row["predicted_at"]),
                symbol=symbol,
                side=OrderSide.SHORT,
                qty=qty,
                signal_price=current_price,
                order_price=order_price,
                order_type=order_type,
                order_result=result,
                broker=broker,
                mode=mode,
                order_session=order_session,
                split_ratio=short_split_ratio,
                horizon=_determine_entry_horizon(row),
                trade_diff_sink=trade_diff_sink,
            )
            logger.info(
                f"[exec] ショート発注: {symbol} {qty}株 @ {order_type.name}({order_reason}) "
                f"(1d変化率={row['diff_ratio']:.3%}, 統合スコア={row['multi_horizon_score']:.3%}, "
                f"閾値={row['effective_sell_threshold']:.3%}, split={short_split_ratio:.0%})"
            )
            stats["short_orders"] += 1
            stats["total_turnover"] += current_price * qty
            position_count += 1
            sent += 1
        except Exception as e:
            logger.error(f"[exec] ショート注文エラー ({symbol}): {e}", exc_info=True)
            stats["errors"] += 1
    return position_count
