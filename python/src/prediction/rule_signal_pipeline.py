"""
ルールベース日次シグナルパイプライン

「最優秀ルール」テーブルから有効な銘柄を取得し、
当日の終値データにルールを適用して Buy/Sell/Hold シグナルを生成する。
生成結果は DuckDB に保存し、ペーパートレードに渡す。
"""

from __future__ import annotations

from datetime import date
from typing import Any

import src.market_data.yf_client as yf_client
from src.backtest.rules import (
    BollingerBandRule,
    EMAMomentumRule,
    MACDRSIRule,
    RSIContrarianRule,
    TradingRule,
    VolatilityBreakoutRule,
    VolumeBreakoutRule,
)
from src.market_data.technical import add_technical_indicators
from src.utils.data_path_utils import get_ticker
from src.utils.db.rule_results import load_effective_rules, upsert_rule_signal
from src.utils.logger import get_logger

logger = get_logger(__name__)

_RULE_INSTANCES: dict[str, TradingRule] = {
    "volume_breakout": VolumeBreakoutRule(),
    "ema_momentum": EMAMomentumRule(),
    "rsi_contrarian": RSIContrarianRule(),
    "bollinger_band": BollingerBandRule(),
    "macd_rsi": MACDRSIRule(),
    "volatility_breakout": VolatilityBreakoutRule(),
}

_LOOKBACK_DAYS = 90


def _get_today_signal(
    market: str,
    symbol: str,
    rule_name: str,
) -> tuple[int, float | None]:
    """
    当日のシグナルと終値を返す。

    Returns:
        (signal, price): signal は 1=buy, -1=sell, 0=hold
    """
    rule = _RULE_INSTANCES.get(rule_name)
    if rule is None:
        logger.warning(f"未知のルール: {rule_name}")
        return 0, None

    ticker = get_ticker(market, symbol)
    df = yf_client.download(ticker, period=f"{_LOOKBACK_DAYS}d")
    if df is None or df.empty or len(df) < 20:
        logger.warning(f"データ不足: {ticker}")
        return 0, None

    df = add_technical_indicators(df)
    signal_series = rule.generate_signal(df)

    today_signal = int(signal_series.iloc[-1])
    today_price = float(df["Close"].iloc[-1])
    return today_signal, today_price


def run_rule_signal_pipeline(
    market: str,
    min_win_rate: float = 0.5,
    min_net_profit: float = 0.0,
    signal_date: date | None = None,
) -> list[dict[str, Any]]:
    """
    有効な最優秀ルールを持つ全銘柄について当日シグナルを生成・保存する。

    Args:
        market: マーケット識別子
        min_win_rate: 有効ルールの最低勝率
        min_net_profit: 有効ルールの最低純利益
        signal_date: シグナル日付（Noneで今日）

    Returns:
        各銘柄のシグナル情報リスト
    """
    if signal_date is None:
        signal_date = date.today()

    effective_df = load_effective_rules(market, min_win_rate, min_net_profit)
    if effective_df.empty:
        logger.warning("有効なルールを持つ銘柄がありません（週次評価を先に実行してください）")
        return []

    logger.info(f"シグナル生成: {len(effective_df)} 銘柄 ({market})")
    results = []

    for _, row in effective_df.iterrows():
        symbol = row["symbol"]
        rule_name = row["best_rule"]
        logger.info(f"  {symbol} → [{rule_name}] シグナル計算中...")

        try:
            signal, price = _get_today_signal(market, symbol, rule_name)
            upsert_rule_signal(signal_date, market, symbol, rule_name, signal, price)

            label = {1: "BUY", -1: "SELL", 0: "HOLD"}.get(signal, "HOLD")
            results.append(
                {
                    "symbol": symbol,
                    "rule": rule_name,
                    "signal": signal,
                    "signal_label": label,
                    "price": price,
                    "win_rate": row["win_rate"],
                    "net_profit": row["net_profit"],
                }
            )
            logger.info(f"  {symbol}: {label}  価格={price}")

        except Exception as exc:
            logger.error(f"  {symbol} シグナル生成失敗: {exc}", exc_info=True)

    return results


def execute_rule_paper_trades(
    signals: list[dict[str, Any]],
    market: str,
    initial_budget_per_trade: float = 100_000,
) -> dict[str, int]:
    """
    シグナルに基づきペーパートレードを実行する。

    - BUY signal  → ポジション未保有の場合のみ買い注文
    - SELL signal → ポジション保有中の場合のみ売り注文

    Args:
        signals: run_rule_signal_pipeline の返り値
        market: マーケット識別子
        initial_budget_per_trade: 1銘柄あたりの最大投資額（概算）

    Returns:
        {"buy_orders": int, "sell_orders": int, "skipped": int}
    """
    from src.trading.brokers.base import OrderSide, OrderType
    from src.trading.brokers.paper.paper_broker import PaperBroker
    from src.utils.data_path_utils import get_ticker

    broker = PaperBroker()
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
