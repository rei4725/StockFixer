"""
ルールベース日次シグナルパイプライン

「最優秀ルール」テーブルから有効な銘柄を取得し、
当日の終値データにルールを適用して Buy/Sell/Hold シグナルを生成する。
生成結果は DuckDB に保存する。
"""

from __future__ import annotations

from datetime import date
from typing import Any

from src.domain.generated_rules import GENERATED_RULES
from src.rule_engine.ports import OHLCVWithIndicatorsPort
from src.rule_engine.rules import (
    BollingerBandRule,
    EMAMomentumRule,
    MACDRSIRule,
    RSIContrarianRule,
    TradingRule,
    VolatilityBreakoutRule,
    VolumeBreakoutRule,
)
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
    **GENERATED_RULES,
}

_LOOKBACK_DAYS = 90


def _get_today_signal(
    market: str,
    symbol: str,
    rule_name: str,
    market_data_port: OHLCVWithIndicatorsPort,
) -> tuple[int, float | None]:
    """当日のシグナルと終値を返す。Returns: (signal, price) signal は 1=buy, -1=sell, 0=hold"""
    rule = _RULE_INSTANCES.get(rule_name)
    if rule is None:
        logger.warning(f"未知のルール: {rule_name}")
        return 0, None

    ticker = get_ticker(market, symbol)
    df = market_data_port.get_ohlcv_with_indicators(ticker, _LOOKBACK_DAYS)
    if df is None or df.empty or len(df) < 20:
        logger.warning(f"データ不足: {ticker}")
        return 0, None

    signal_series = rule.generate_signal(df)

    today_signal = int(signal_series.iloc[-1])
    today_price = float(df["Close"].iloc[-1])
    return today_signal, today_price


def run_rule_signal_pipeline(
    market: str,
    min_win_rate: float = 0.5,
    min_net_profit: float = 0.0,
    signal_date: date | None = None,
    market_data_port: OHLCVWithIndicatorsPort | None = None,
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
    if market_data_port is None:
        raise ValueError(
            "market_data_port は必須です。呼び出し元で OHLCVWithIndicatorsPort 実装を注入してください。"
        )

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
            signal, price = _get_today_signal(market, symbol, rule_name, market_data_port)
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
