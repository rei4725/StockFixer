"""レジームレバレッジ戦略(TQQQ/短期債と同様の自己完結モジュール)の判定ロジック。

STRATEGY.md 7章の週次(レジーム転換・新規エントリー)・日次(初期損切り・マージンコール)
判定を実装する。バックテスト(trading-strategy/backtest/backtest_regime_leverage.py)と
同じ計算式を使う。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from config.settings import (
    REGIME_LEVERAGE_INITIAL_STOP_ATR_MULT,
    REGIME_LEVERAGE_INTEREST_ANNUAL,
    REGIME_LEVERAGE_MARGIN_MAINTENANCE,
    REGIME_LEVERAGE_RATIO,
    REGIME_LEVERAGE_SLIPPAGE_PCT,
)
from src.domain.ports import MarketDataPort
from src.trading.regime_leverage_strategy.indicators import build_weekly_frame
from src.trading.regime_leverage_strategy.repository import get_latest_snapshot, insert_snapshot
from src.trading.regime_leverage_strategy.types import (
    RegimeLeverageDecision,
    RegimeLeverageSnapshot,
)

# 米国株信用の手数料(trading-strategy/backtest.pyのCOMMISSION_PCT["USD"]と同じ値)
_COMMISSION_PCT_USD = 0.0033
_COMMISSION_CAP_USD = 16.5


def _calc_commission_usd(notional_usd: float) -> float:
    return min(notional_usd * _COMMISSION_PCT_USD, _COMMISSION_CAP_USD)


def compute_equity_now(
    snapshot: RegimeLeverageSnapshot, current_price_jpy: float, now: datetime
) -> float:
    """保有中ポジションの現在評価額(円)を再計算する。

    バックテストのように保有期間中の含み損益・金利を都度累積して持ち回るのではなく、
    エントリー時点の情報(entry_date/entry_price_jpy/equity_at_entry_jpy/
    entry_commission_jpy)から毎回再計算する(累積値の更新漏れによるバグを避けるため)。
    """
    if (
        snapshot.entry_price_jpy is None
        or snapshot.entry_date is None
        or snapshot.equity_at_entry_jpy is None
    ):
        raise ValueError("保有中でないsnapshotに対してcompute_equity_nowは呼べない")
    unrealized_pnl = (current_price_jpy - snapshot.entry_price_jpy) * snapshot.shares
    days_held = (now.date() - snapshot.entry_date.date()).days
    interest_accrued = (
        snapshot.entry_price_jpy
        * snapshot.shares
        * REGIME_LEVERAGE_INTEREST_ANNUAL
        / 365
        * days_held
    )
    commission = snapshot.entry_commission_jpy or 0.0
    return snapshot.equity_at_entry_jpy + unrealized_pnl - interest_accrued - commission


def _cash_preserved_noop(
    cash_jpy: float, week_close_usd: float, usdjpy_rate: float
) -> RegimeLeverageDecision:
    """未保有・非エントリー時のnoop応答(現金は変化せずそのままequity_now_jpyに反映)。"""
    return RegimeLeverageDecision(
        action="noop",
        reason="weekly_noop",
        spy_price_usd=week_close_usd,
        usdjpy_rate=usdjpy_rate,
        shares=0.0,
        entry_date=None,
        entry_price_jpy=None,
        entry_commission_jpy=None,
        equity_at_entry_jpy=None,
        stop_price_jpy=None,
        equity_now_jpy=cash_jpy,
        maintenance_ratio=None,
    )


def _closed_position_decision(
    reason: str,
    spy_price_usd: float,
    usdjpy_rate: float,
    equity_now_jpy: float,
    maintenance_ratio: Optional[float] = None,
) -> RegimeLeverageDecision:
    """ポジション解消(exit)時の共通レスポンス生成(建玉関連フィールドは全てNone/0にリセットする)。

    regime_flip(週次)・margin_call/initial_stop(日次)の3箇所で同一の
    フィールド構成が必要になるため共通化する。
    """
    return RegimeLeverageDecision(
        action="exit",
        reason=reason,
        spy_price_usd=spy_price_usd,
        usdjpy_rate=usdjpy_rate,
        shares=0.0,
        entry_date=None,
        entry_price_jpy=None,
        entry_commission_jpy=None,
        equity_at_entry_jpy=None,
        stop_price_jpy=None,
        equity_now_jpy=equity_now_jpy,
        maintenance_ratio=maintenance_ratio,
    )


def decide_weekly_entry(
    cash_jpy: float,
    week_close_usd: float,
    ma200_usd: float,
    atr14_usd: float,
    usdjpy_rate: float,
    now: datetime,
) -> RegimeLeverageDecision:
    """未保有時の週次判定: レジームが上昇なら新規エントリーする。"""
    regime_up = week_close_usd > ma200_usd
    if not regime_up:
        return _cash_preserved_noop(cash_jpy, week_close_usd, usdjpy_rate)

    entry_price_usd = week_close_usd * (1 + REGIME_LEVERAGE_SLIPPAGE_PCT)
    entry_price_jpy = entry_price_usd * usdjpy_rate
    notional_target_usd = (cash_jpy / usdjpy_rate) * REGIME_LEVERAGE_RATIO
    shares = float(int(notional_target_usd // entry_price_usd))
    if shares <= 0:
        return _cash_preserved_noop(cash_jpy, week_close_usd, usdjpy_rate)

    commission_jpy = _calc_commission_usd(entry_price_usd * shares) * usdjpy_rate
    stop_price_jpy = (
        week_close_usd - REGIME_LEVERAGE_INITIAL_STOP_ATR_MULT * atr14_usd
    ) * usdjpy_rate

    return RegimeLeverageDecision(
        action="entry",
        reason="regime_entry",
        spy_price_usd=week_close_usd,
        usdjpy_rate=usdjpy_rate,
        shares=shares,
        entry_date=now,
        entry_price_jpy=entry_price_jpy,
        entry_commission_jpy=commission_jpy,
        equity_at_entry_jpy=cash_jpy,
        stop_price_jpy=stop_price_jpy,
        equity_now_jpy=cash_jpy - commission_jpy,
        maintenance_ratio=None,
    )


def decide_weekly_exit(
    snapshot: RegimeLeverageSnapshot,
    week_close_usd: float,
    ma200_usd: float,
    usdjpy_rate: float,
    now: datetime,
) -> RegimeLeverageDecision:
    """保有中の週次判定: レジーム転換のみ判定する(初期損切り・マージンコールは日次ジョブが担当)。"""
    current_price_jpy = week_close_usd * usdjpy_rate
    equity_now = compute_equity_now(snapshot, current_price_jpy, now)
    regime_up = week_close_usd > ma200_usd

    if regime_up:
        return RegimeLeverageDecision(
            action="noop",
            reason="weekly_noop",
            spy_price_usd=week_close_usd,
            usdjpy_rate=usdjpy_rate,
            shares=snapshot.shares,
            entry_date=snapshot.entry_date,
            entry_price_jpy=snapshot.entry_price_jpy,
            entry_commission_jpy=snapshot.entry_commission_jpy,
            equity_at_entry_jpy=snapshot.equity_at_entry_jpy,
            stop_price_jpy=snapshot.stop_price_jpy,
            equity_now_jpy=equity_now,
            maintenance_ratio=None,
        )

    exit_price_jpy = current_price_jpy * (1 - REGIME_LEVERAGE_SLIPPAGE_PCT)
    exit_equity = compute_equity_now(snapshot, exit_price_jpy, now)
    return _closed_position_decision(
        reason="regime_flip",
        spy_price_usd=week_close_usd,
        usdjpy_rate=usdjpy_rate,
        equity_now_jpy=exit_equity,
    )


def decide_daily_check(
    snapshot: RegimeLeverageSnapshot,
    day_low_usd: float,
    usdjpy_rate: float,
    now: datetime,
) -> RegimeLeverageDecision:
    """保有中の日次判定: マージンコール→初期損切りの優先順位で当日安値ベースに判定する
    (バックテストのrun_levered_regimeと同じ優先順位)。
    """
    day_low_jpy = day_low_usd * usdjpy_rate
    equity_at_low = compute_equity_now(snapshot, day_low_jpy, now)
    value_at_low = day_low_jpy * snapshot.shares
    maintenance_ratio = equity_at_low / value_at_low if value_at_low > 0 else 0.0

    if value_at_low > 0 and maintenance_ratio < REGIME_LEVERAGE_MARGIN_MAINTENANCE:
        exit_price_jpy = day_low_jpy * (1 - REGIME_LEVERAGE_SLIPPAGE_PCT)
        exit_equity = compute_equity_now(snapshot, exit_price_jpy, now)
        return _closed_position_decision(
            reason="margin_call",
            spy_price_usd=day_low_usd,
            usdjpy_rate=usdjpy_rate,
            equity_now_jpy=exit_equity,
            maintenance_ratio=maintenance_ratio,
        )

    if snapshot.stop_price_jpy is not None and day_low_jpy <= snapshot.stop_price_jpy:
        exit_price_jpy = snapshot.stop_price_jpy * (1 - REGIME_LEVERAGE_SLIPPAGE_PCT)
        exit_equity = compute_equity_now(snapshot, exit_price_jpy, now)
        return _closed_position_decision(
            reason="initial_stop",
            spy_price_usd=day_low_usd,
            usdjpy_rate=usdjpy_rate,
            equity_now_jpy=exit_equity,
            maintenance_ratio=maintenance_ratio,
        )

    return RegimeLeverageDecision(
        action="noop",
        reason="daily_noop",
        spy_price_usd=day_low_usd,
        usdjpy_rate=usdjpy_rate,
        shares=snapshot.shares,
        entry_date=snapshot.entry_date,
        entry_price_jpy=snapshot.entry_price_jpy,
        entry_commission_jpy=snapshot.entry_commission_jpy,
        equity_at_entry_jpy=snapshot.equity_at_entry_jpy,
        stop_price_jpy=snapshot.stop_price_jpy,
        equity_now_jpy=equity_at_low,
        maintenance_ratio=maintenance_ratio,
    )


def _load_spy_daily(market_data_port: MarketDataPort, symbol: str) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=400)  # 200日線を計算するため十分な余裕を持って取得
    df = market_data_port.get_stock_data(
        symbol, "us", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    )
    return build_weekly_frame(df)


def _load_latest_usdjpy(market_data_port: MarketDataPort) -> float:
    end = datetime.now()
    start = end - timedelta(days=10)
    fx_df = market_data_port.get_forex_data(
        "JPY=X", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    )
    return float(fx_df["Close"].iloc[-1])


def run_regime_leverage_weekly_check(market_data_port: MarketDataPort) -> RegimeLeverageDecision:
    """週次ジョブ本体: レジーム転換の判定、または未保有時の新規エントリー判定を行い、
    結果をDBに記録して返す。
    """
    from config.settings import REGIME_LEVERAGE_SYMBOL

    df = _load_spy_daily(market_data_port, REGIME_LEVERAGE_SYMBOL)
    usdjpy_rate = _load_latest_usdjpy(market_data_port)
    latest_row = df.iloc[-1]
    week_close_usd = float(latest_row["Close"])
    ma200_usd = float(latest_row["MA200"])
    atr14_usd = float(latest_row["ATR14"])
    now = datetime.now()

    snapshot = get_latest_snapshot()

    if snapshot is None or snapshot.shares <= 0:
        from config.settings import REGIME_LEVERAGE_INITIAL_CAPITAL_JPY

        cash_jpy = (
            snapshot.equity_now_jpy if snapshot is not None else REGIME_LEVERAGE_INITIAL_CAPITAL_JPY
        )
        decision = decide_weekly_entry(
            cash_jpy, week_close_usd, ma200_usd, atr14_usd, usdjpy_rate, now
        )
    else:
        decision = decide_weekly_exit(snapshot, week_close_usd, ma200_usd, usdjpy_rate, now)

    insert_snapshot(decision)
    return decision


def run_regime_leverage_daily_margin_check(
    market_data_port: MarketDataPort,
) -> Optional[RegimeLeverageDecision]:
    """日次ジョブ本体: 保有中の場合のみ、初期損切り・マージンコールを判定する。"""
    from config.settings import REGIME_LEVERAGE_SYMBOL

    snapshot = get_latest_snapshot()
    if snapshot is None or snapshot.shares <= 0:
        return None

    df = _load_spy_daily(market_data_port, REGIME_LEVERAGE_SYMBOL)
    usdjpy_rate = _load_latest_usdjpy(market_data_port)
    day_low_usd = float(df.iloc[-1]["Low"])
    now = datetime.now()

    decision = decide_daily_check(snapshot, day_low_usd, usdjpy_rate, now)
    insert_snapshot(decision)
    return decision
