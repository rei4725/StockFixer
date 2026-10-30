"""
ルールベース日次シグナルパイプライン

「最優秀ルール」テーブルから有効な銘柄を取得し、
当日の終値データにルールを適用して Buy/Sell/Hold シグナルを生成する。
生成結果は DuckDB に保存する。
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

import pandas as pd

from src.domain.generated_rules import GENERATED_RULES
from src.domain.jp_market_holidays import is_jp_business_day
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
from src.rule_engine.types import RuleSignalBatch, UnavailableRuleSymbol
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

# 最後の足が signal_date の直近の営業日より平日でこの日数を超えて古ければ、足が止まったとみなす。
# PR 828 の config.settings.STALE_BARS_MAX_WEEKDAYS と同じ値。828 のマージ後に1つにまとめる
_STALE_BARS_MAX_WEEKDAYS = 3


def _is_open_day(day: date, market: str) -> bool:
    """jp は東証の営業日（#1164）、それ以外は土日以外なら True。"""
    if market == "jp":
        return is_jp_business_day(day)
    return day.weekday() < 5


def _count_weekdays_behind(last_bar: date, signal_date: date, market: str = "us") -> int:
    """最後の足の翌日から、signal_date 以前のあるべき最新の営業日までの営業日の数を返す。

    数え方は prediction.predict_unified.count_weekdays_behind（#1106）に揃える。
    jp は東証の休場日も除き、それ以外は土日だけを除く。
    """
    expected_latest = signal_date
    while not _is_open_day(expected_latest, market):
        expected_latest -= timedelta(days=1)
    if last_bar >= expected_latest:
        return 0
    days = pd.date_range(last_bar + timedelta(days=1), expected_latest)
    return sum(1 for day in days if _is_open_day(day.date(), market))


def _get_today_signal(
    market: str,
    symbol: str,
    rule_name: str,
    market_data_port: OHLCVWithIndicatorsPort,
    signal_date: date | None = None,
) -> tuple[int, float] | None:
    """当日のシグナルと終値を返す。Returns: (signal, price) signal は 1=buy, -1=sell, 0=hold

    未知のルール・データ不足・最後の足の終値が有限の正の数でない（#1409）、
    または signal_date を渡したときに最後の足が古すぎれば
    None を返す（シグナルを出さない。HOLD とは分ける #822）。
    """
    rule = _RULE_INSTANCES.get(rule_name)
    if rule is None:
        logger.warning(f"未知のルール: {rule_name}")
        return None

    ticker = get_ticker(market, symbol)
    df = market_data_port.get_ohlcv_with_indicators(ticker, _LOOKBACK_DAYS)
    if df is None or df.empty or len(df) < 20:
        logger.warning(f"データ不足: {ticker}")
        return None

    if signal_date is not None and isinstance(df.index, pd.DatetimeIndex):
        last_bar = pd.Timestamp(df.index[-1]).date()
        behind = _count_weekdays_behind(last_bar, signal_date, market)
        if behind > _STALE_BARS_MAX_WEEKDAYS:
            logger.warning(
                f"足が止まったためシグナルを出さない: {market}/{symbol} "
                f"（最後の足 {last_bar}、平日で {behind}日 > {_STALE_BARS_MAX_WEEKDAYS}日 古い。"
                "取得の失敗か上場廃止の可能性）"
            )
            return None

    # 終値だけ欠けた足は取得の後も残る。HOLD や NaN の値段で渡さず、出せない銘柄にする（#1409）
    today_price = float(pd.to_numeric(df["Close"].iloc[-1:], errors="coerce").iloc[0])
    if not math.isfinite(today_price) or today_price <= 0:
        logger.warning(
            f"最後の足の終値が使えないためシグナルを出さない: {market}/{symbol} "
            f"（終値 {today_price}）"
        )
        return None

    signal_series = rule.generate_signal(df)

    today_signal = int(signal_series.iloc[-1])
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
        各銘柄のシグナル情報リスト（シグナルを出せなかった銘柄は含まない）
    """
    return generate_rule_signals(
        market, min_win_rate, min_net_profit, signal_date, market_data_port
    ).signals


def generate_rule_signals(
    market: str,
    min_win_rate: float = 0.5,
    min_net_profit: float = 0.0,
    signal_date: date | None = None,
    market_data_port: OHLCVWithIndicatorsPort | None = None,
) -> RuleSignalBatch:
    """
    run_rule_signal_pipeline と同じくシグナルを生成・保存し、出せなかった銘柄も返す（#822）。

    データ不足・未知のルール・足の停止・例外の銘柄は HOLD として保存せず、
    unavailable に入れる。total は対象の銘柄数。
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
        return RuleSignalBatch()

    logger.info(f"シグナル生成: {len(effective_df)} 銘柄 ({market})")
    results = []
    unavailable: list[UnavailableRuleSymbol] = []

    for _, row in effective_df.iterrows():
        symbol = row["symbol"]
        rule_name = row["best_rule"]
        logger.info(f"  {symbol} → [{rule_name}] シグナル計算中...")

        try:
            today = _get_today_signal(market, symbol, rule_name, market_data_port, signal_date)
            if today is None:
                unavailable.append(UnavailableRuleSymbol(symbol, rule_name, "シグナルなし"))
                continue
            signal, price = today
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
            unavailable.append(UnavailableRuleSymbol(symbol, rule_name, "例外"))

    return RuleSignalBatch(signals=results, unavailable=unavailable, total=len(effective_df))
