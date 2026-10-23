"""レジームレバレッジ戦略(TQQQ/短期債と同様の自己完結モジュール)の判定ロジック。

STRATEGY.md 7章の週次(レジーム転換・新規エントリー)・日次(初期損切り・マージンコール)
判定を実装する。バックテスト(trading-strategy/backtest/backtest_regime_leverage.py)と
同じ計算式を使う。
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

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
from src.trading.regime_leverage_strategy.repository import (
    get_last_daily_check_at,
    get_latest_snapshot,
    insert_snapshot,
)
from src.trading.regime_leverage_strategy.types import (
    RegimeLeverageDecision,
    RegimeLeverageSnapshot,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 米国株信用の手数料(trading-strategy/backtest.pyのCOMMISSION_PCT["USD"]と同じ値)
_COMMISSION_PCT_USD = 0.0033
_COMMISSION_CAP_USD = 16.5

_JST = ZoneInfo("Asia/Tokyo")
_US_MARKET_TZ = ZoneInfo("America/New_York")
_US_MARKET_CLOSE = time(16, 0)


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


def _bar_close_at(bar_date: object) -> datetime:
    """日足の引けの時刻(米国の取引所の16:00、夏時間・冬時間は ZoneInfo が扱う)を返す。"""
    day = pd.Timestamp(bar_date).date()
    return datetime.combine(day, _US_MARKET_CLOSE, tzinfo=_US_MARKET_TZ)


def _as_aware(dt: datetime) -> datetime:
    """タイムゾーンの無い時刻は JST とみなす(スケジューラーは JST で動く)。"""
    return dt.replace(tzinfo=_JST) if dt.tzinfo is None else dt


def select_unchecked_closed_bars(
    df: pd.DataFrame, last_checked_at: Optional[datetime], now: datetime
) -> pd.DataFrame:
    """日次判定で見る日足を返す: now までに引けた足のうち、前の回(last_checked_at)の後に
    引けた足を全部。引けていない当日の足は含めない。前の回の後に引けた足が無ければ、
    最後に引けた足だけを返す(#891)。
    """
    now_aware = _as_aware(now)
    closes = [_bar_close_at(ts) for ts in df.index]
    closed = df[[close_at <= now_aware for close_at in closes]]
    if closed.empty:
        return closed
    if last_checked_at is not None:
        since = _as_aware(last_checked_at)
        unchecked = closed[[_bar_close_at(ts) > since for ts in closed.index]]
        if not unchecked.empty:
            return unchecked
    return closed.iloc[[-1]]


def _load_latest_usdjpy(market_data_port: MarketDataPort) -> float:
    """最新のドル円(終値)を返す。

    最終行の終値がNaNまたは0以下(データ劣化時)の場合はRuntimeErrorを送出する
    (NaN比較は常にFalseになり、マージンコール・損切りの判定が黙って飛んでNaNの
    評価額が記録されるため。スケジューラーのリトライ・Discord通知に乗せる。#871)。
    """
    end = datetime.now()
    start = end - timedelta(days=10)
    fx_df = market_data_port.get_forex_data(
        "JPY=X", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    )
    usdjpy_rate = float(fx_df["Close"].iloc[-1])
    if pd.isna(usdjpy_rate) or usdjpy_rate <= 0:
        raise RuntimeError(
            "レジームレバレッジ戦略: ドル円の最終行の終値が不正のため判定を中止しました"
            f"(usdjpy_rate={usdjpy_rate})"
        )
    return usdjpy_rate


def run_regime_leverage_weekly_check(
    market_data_port: MarketDataPort,
) -> Optional[RegimeLeverageDecision]:
    """週次ジョブ本体: レジーム転換の判定、または未保有時の新規エントリー判定を行い、
    結果をDBに記録して返す。

    MA200/ATR14がNaN(取得データが200/14日分に満たない等、データ劣化時)の場合は
    判定を行わない(NaN比較は常にFalseになるため、誤って「レジーム下降」と
    解釈して保有ポジションを強制決済してしまうことを防ぐ)。未保有ならNoneを返し、
    保有中ならRuntimeErrorを送出する(レジーム転換の判定が飛んだことを
    スケジューラーのリトライ・Discord通知に乗せるため。#870)。
    """
    from config.settings import REGIME_LEVERAGE_SYMBOL

    df = _load_spy_daily(market_data_port, REGIME_LEVERAGE_SYMBOL)
    usdjpy_rate = _load_latest_usdjpy(market_data_port)
    latest_row = df.iloc[-1]
    week_close_usd = float(latest_row["Close"])
    ma200_usd = float(latest_row["MA200"])
    atr14_usd = float(latest_row["ATR14"])
    snapshot = get_latest_snapshot()
    if pd.isna(ma200_usd) or pd.isna(atr14_usd):
        if snapshot is not None and snapshot.shares > 0:
            raise RuntimeError(
                "レジームレバレッジ戦略(週次): 保有中にMA200またはATR14がNaNのため判定を中止しました"
                f"(ma200_usd={ma200_usd}, atr14_usd={atr14_usd}, shares={snapshot.shares})"
            )
        logger.warning(
            "レジームレバレッジ戦略(週次): MA200またはATR14がNaNのため判定を中止します"
            "(ma200_usd=%s, atr14_usd=%s)",
            ma200_usd,
            atr14_usd,
        )
        return None
    now = datetime.now()

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
    """日次ジョブ本体: 保有中の場合のみ、初期損切り・マージンコールを判定する。

    未保有の場合はNoneを返す(何もしない)。保有中に当日安値(Low)がNaN(データ劣化時)の
    場合は判定を行わずRuntimeErrorを送出する(損切り・マージンコールの判定が飛んだことを
    スケジューラーのリトライ・Discord通知に乗せるため。#860)。
    """
    from config.settings import REGIME_LEVERAGE_SYMBOL

    snapshot = get_latest_snapshot()
    if snapshot is None or snapshot.shares <= 0:
        return None

    df = _load_spy_daily(market_data_port, REGIME_LEVERAGE_SYMBOL)
    usdjpy_rate = _load_latest_usdjpy(market_data_port)
    now = datetime.now(_JST)
    # 前の回で判定した時刻は、日次の回の行(daily_noop)か建てた行で覚える。週次の回は損切りを
    # 判定しないため、週次の行の時刻を使うとその前に引けた足(金曜の足など)を飛ばす。最新の行が
    # 週次の行なら、その前の日次の行を読み、その後に落ちた回の足も判定する(#933)
    if snapshot.reason == "daily_noop":
        last_checked_at: Optional[datetime] = snapshot.executed_at
    else:
        last_checked_at = get_last_daily_check_at()
    bars = select_unchecked_closed_bars(df, last_checked_at, now)
    if bars.empty:
        raise RuntimeError(
            "レジームレバレッジ戦略(日次): 保有中に引けた日足が無いため判定を中止しました"
            f"(shares={snapshot.shares})"
        )

    # 前の回の後に引けた足を古い順に判定し、最初に決済となった足で決済する(#891)
    day_lows_usd = [float(low) for low in bars["Low"]]
    if any(pd.isna(low) for low in day_lows_usd):
        raise RuntimeError(
            "レジームレバレッジ戦略(日次): 保有中に当日安値(Low)がNaNのため判定を中止しました"
            f"(shares={snapshot.shares})"
        )
    decisions = (decide_daily_check(snapshot, low, usdjpy_rate, now) for low in day_lows_usd)
    decision = next((d for d in decisions if d.action == "exit"), None)
    if decision is None:
        decision = decide_daily_check(snapshot, day_lows_usd[-1], usdjpy_rate, now)
    insert_snapshot(decision)
    return decision
