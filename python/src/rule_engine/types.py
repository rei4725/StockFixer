"""rule_engine BC の型定義"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional


@dataclass
class RuleSignal:
    symbol: str
    rule: str
    signal: int  # 1=buy, -1=sell, 0=hold
    signal_label: str
    price: Optional[float]
    win_rate: float
    net_profit: float


@dataclass
class RuleResult:
    market: str
    signal_date: date
    signals: list[RuleSignal] = field(default_factory=list)


@dataclass
class UnavailableRuleSymbol:
    """シグナルを出せなかった銘柄（データ不足・未知のルール・足の停止・例外）。HOLD とは別に数える。"""

    symbol: str
    rule: str
    reason: str


@dataclass
class RuleSignalBatch:
    """日次ルールシグナルの1回分。signals は保存した銘柄だけで、unavailable は保存していない。"""

    signals: list[dict[str, Any]] = field(default_factory=list)
    unavailable: list[UnavailableRuleSymbol] = field(default_factory=list)
    total: int = 0
