"""
trading BC の型定義。

発注・リスク管理で使用するドメイン型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CorrelationGateResult:
    """相関ベースのポートフォリオリスクゲートの評価結果。"""

    is_allowed: bool
    enc: float
    enc_threshold: float
    avg_correlation: float
    n_symbols: int
    symbols: list[str] = field(default_factory=list)
    reason: Optional[str] = None


@dataclass
class TradingGateStatus:
    """発注前リスクゲートの評価結果。"""

    is_allowed: bool
    stop_active: bool
    reason_code: Optional[str] = None
    reason: Optional[str] = None
    daily_loss: float = 0.0
    daily_loss_limit: Optional[float] = None
    consecutive_losses: int = 0
    consecutive_loss_limit: Optional[int] = None
    position_count: int = 0
    max_positions: Optional[int] = None
