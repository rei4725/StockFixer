"""rule_engine BC の型定義"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


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
