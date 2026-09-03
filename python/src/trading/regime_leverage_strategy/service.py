"""レジームレバレッジ戦略(TQQQ/短期債と同様の自己完結モジュール)の判定ロジック。

STRATEGY.md 7章の週次(レジーム転換・新規エントリー)・日次(初期損切り・マージンコール)
判定を実装する。バックテスト(trading-strategy/backtest/backtest_regime_leverage.py)と
同じ計算式を使う。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional  # noqa: F401  # Task 5/6の型ヒントで使用予定

from config.settings import (  # noqa: F401  # Task 5のマージンコール判定で使用予定
    REGIME_LEVERAGE_INITIAL_STOP_ATR_MULT,
    REGIME_LEVERAGE_INTEREST_ANNUAL,
    REGIME_LEVERAGE_MARGIN_MAINTENANCE,
    REGIME_LEVERAGE_RATIO,
    REGIME_LEVERAGE_SLIPPAGE_PCT,
)
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
    return RegimeLeverageDecision(
        action="exit",
        reason="regime_flip",
        spy_price_usd=week_close_usd,
        usdjpy_rate=usdjpy_rate,
        shares=0.0,
        entry_date=None,
        entry_price_jpy=None,
        entry_commission_jpy=None,
        equity_at_entry_jpy=None,
        stop_price_jpy=None,
        equity_now_jpy=exit_equity,
        maintenance_ratio=None,
    )
