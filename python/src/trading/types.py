"""
trading BC の型定義。

発注・リスク管理で使用するドメイン型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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
