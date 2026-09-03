"""レジームレバレッジ戦略(TQQQ/短期債と同様の自己完結モジュール)の型定義。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class RegimeLeverageSnapshot:
    """ある時点でのレジームレバレッジ戦略の建玉・評価額状態(regime_leverage_logの1行)。"""

    id: int
    executed_at: datetime
    action: str
    reason: str
    spy_price_usd: float
    usdjpy_rate: float
    shares: float
    entry_date: Optional[datetime]
    entry_price_jpy: Optional[float]
    entry_commission_jpy: Optional[float]
    equity_at_entry_jpy: Optional[float]
    stop_price_jpy: Optional[float]
    equity_now_jpy: float
    maintenance_ratio: Optional[float]


@dataclass(frozen=True)
class RegimeLeverageDecision:
    """週次/日次の判定結果(insert_snapshotへそのまま渡せる形)。"""

    action: str  # 'entry' | 'exit' | 'noop'
    reason: str  # noqa: E501  # 'regime_entry' | 'regime_flip' | 'initial_stop' | 'margin_call' | 'weekly_noop' | 'daily_noop'
    spy_price_usd: float
    usdjpy_rate: float
    shares: float
    entry_date: Optional[datetime]
    entry_price_jpy: Optional[float]
    entry_commission_jpy: Optional[float]
    equity_at_entry_jpy: Optional[float]
    stop_price_jpy: Optional[float]
    equity_now_jpy: float
    maintenance_ratio: Optional[float]
