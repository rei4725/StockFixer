"""配分戦略(TQQQ/短期債)ペーパートレードの型定義。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class AllocationSnapshot:
    """ある時点での配分戦略の建玉・現金状態(allocation_rebalance_logの1行)。"""

    id: int
    executed_at: datetime
    action: str
    tqqq_price: float
    shy_price: float
    tqqq_qty_before: float
    shy_qty_before: float
    cash_before: float
    tqqq_qty_after: float
    shy_qty_after: float
    cash_after: float


@dataclass(frozen=True)
class RebalanceOutcome:
    """run_allocation_rebalance() が実行した結果(Discord通知に使う)。"""

    action: str
    tqqq_price: float
    shy_price: float
    tqqq_qty_before: float
    shy_qty_before: float
    cash_before: float
    tqqq_qty_after: float
    shy_qty_after: float
    cash_after: float
