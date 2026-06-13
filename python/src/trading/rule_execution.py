"""ルールシグナルに基づくペーパートレード実行"""

from __future__ import annotations

from typing import Any

from config.settings import MAX_ORDERS_PER_RUN, MAX_POSITIONS
from src.trading.brokers.base import OrderSide, OrderType
from src.trading.brokers.paper.paper_broker import PaperBroker
from src.trading.risk_manager import RiskManager
from src.utils.data_path_utils import get_ticker
from src.utils.logger import get_logger

logger = get_logger(__name__)


def execute_rule_paper_trades(
    signals: list[dict[str, Any]],
    market: str,
    initial_budget_per_trade: float = 100_000,
    market_data_port=None,
) -> dict[str, int]:
    """
    シグナルに基づきペーパートレードを実行する。

    - BUY signal  → ポジション未保有の場合のみ買い注文
    - SELL signal → ポジション保有中の場合のみ売り注文

    Args:
        signals: run_rule_signal_pipeline の返り値
        market: マーケット識別子
        initial_budget_per_trade: 1銘柄あたりの最大投資額（概算）
        market_data_port: MarketDataPort 実装（呼び出し元から注入）

    Returns:
        {"buy_orders": int, "sell_orders": int, "skipped": int}
    """
    broker = PaperBroker(market_data_port=market_data_port)
    risk = RiskManager(broker)
    buy_orders = 0
    sell_orders = 0
    skipped = 0

    try:
        positions = broker.get_positions()
        held_symbols = {p["symbol"] for p in positions}
    except Exception:
        positions = []
        held_symbols = set()

    # ML 経路（execution.run_daily_orders）と同等のリスク制御を適用する。
    # R-307: DD 基準値を発注前に更新し、当日発注可否ゲート（日次損失・連続損失・
    # 保有上限）を評価する。不合格時は新規買いを停止するが、決済（SELL）は許可する。
    risk.update_peak_balance()
    gate = risk.evaluate_trading_gate()
    if not gate.is_allowed:
        logger.warning(
            "[rule] リスクゲート不合格のため新規買いを停止: %s",
            gate.reason or "理由未設定",
        )

    position_count = len(held_symbols)  # 1ラン内で増える建玉数を追跡し MAX_POSITIONS を厳守
    buy_orders_this_run = 0

    for item in signals:
        symbol = item["symbol"]
        signal = item["signal"]
        price = item.get("price") or 0.0

        try:
            if signal == 1 and symbol not in held_symbols:
                if not gate.is_allowed:
                    skipped += 1
                    continue
                if position_count >= MAX_POSITIONS:
                    logger.info(
                        "[rule] %s: 保有上限 %d 到達のため新規買いスキップ", symbol, MAX_POSITIONS
                    )
                    skipped += 1
                    continue
                if buy_orders_this_run >= MAX_ORDERS_PER_RUN:
                    logger.info(
                        "[rule] %s: 1ラン発注上限 %d 到達のため新規買いスキップ",
                        symbol,
                        MAX_ORDERS_PER_RUN,
                    )
                    skipped += 1
                    continue

                # 残高連動のハーフ Kelly 基準で株数を算出（残高不足時は 0 株 → スキップ）。
                # initial_budget_per_trade は 1 銘柄あたりの投資上限として併用する（保守側）。
                qty = risk.calc_position_size(symbol, price)
                if price > 0 and initial_budget_per_trade > 0:
                    budget_cap = (int(initial_budget_per_trade / price) // 100) * 100
                    qty = min(qty, budget_cap)
                if qty <= 0:
                    logger.info("[rule] %s: 発注株数 0（残高不足 or サイズ上限）→ スキップ", symbol)
                    skipped += 1
                    continue

                ticker = get_ticker(market, symbol)
                broker.send_order(
                    symbol=ticker,
                    side=OrderSide.BUY,
                    qty=qty,
                    order_type=OrderType.MARKET,
                )
                buy_orders += 1
                buy_orders_this_run += 1
                position_count += 1
                logger.info(f"  BUY 注文: {symbol}  {qty}株  現在値={price:.0f}円")

            elif signal == -1 and symbol in held_symbols:
                pos = next((p for p in positions if p["symbol"] == symbol), None)
                qty = pos.get("qty", 1) if pos else 1
                ticker = get_ticker(market, symbol)
                broker.send_order(
                    symbol=ticker,
                    side=OrderSide.SELL,
                    qty=qty,
                    order_type=OrderType.MARKET,
                )
                sell_orders += 1
                logger.info(f"  SELL 注文: {symbol}  {qty}株  現在値={price:.0f}円")

            else:
                skipped += 1

        except Exception as exc:
            logger.error(f"  {symbol} 注文失敗: {exc}", exc_info=True)
            skipped += 1

    logger.info(f"ペーパートレード実行: BUY={buy_orders}  SELL={sell_orders}  スキップ={skipped}")
    return {"buy_orders": buy_orders, "sell_orders": sell_orders, "skipped": skipped}
