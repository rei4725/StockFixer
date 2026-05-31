"""ルールシグナルに基づくペーパートレード実行"""

from __future__ import annotations

from typing import Any

from src.trading.brokers.base import OrderSide, OrderType
from src.trading.brokers.paper.paper_broker import PaperBroker
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
    buy_orders = 0
    sell_orders = 0
    skipped = 0

    try:
        positions = broker.get_positions()
        held_symbols = {p["symbol"] for p in positions}
    except Exception:
        held_symbols = set()

    for item in signals:
        symbol = item["symbol"]
        signal = item["signal"]
        price = item.get("price") or 0.0

        try:
            if signal == 1 and symbol not in held_symbols:
                qty = max(1, int(initial_budget_per_trade / price)) if price > 0 else 1
                ticker = get_ticker(market, symbol)
                broker.send_order(
                    symbol=ticker,
                    side=OrderSide.BUY,
                    qty=qty,
                    order_type=OrderType.MARKET,
                )
                buy_orders += 1
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
