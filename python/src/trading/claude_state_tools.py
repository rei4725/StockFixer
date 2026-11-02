"""Claude の道の tool のうち、保有・残高・リスクの状態を読むだけのハンドラー。

claude_agent.py から分けた（600 行の上限、#1490）。claude_agent から re-export する。
"""

from __future__ import annotations

from typing import Any

from src.trading.brokers.base import BrokerBase
from src.trading.risk_manager import RiskManager
from src.trading.types import TradingGateStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _handle_get_positions(broker: BrokerBase) -> dict[str, Any]:
    try:
        return {"positions": broker.get_positions()}
    except Exception as e:
        logger.error("[claude_agent] get_positions エラー: %s", e, exc_info=True)
        return {"error": str(e)}


def _handle_get_balance(broker: BrokerBase) -> dict[str, Any]:
    try:
        return {"balance": broker.get_balance()}
    except Exception as e:
        logger.error("[claude_agent] get_balance エラー: %s", e, exc_info=True)
        return {"error": str(e)}


def _handle_get_risk_status(risk: RiskManager) -> dict[str, Any]:
    try:
        gate: TradingGateStatus = risk.evaluate_trading_gate()
        return {
            "is_allowed": gate.is_allowed,
            "stop_active": gate.stop_active,
            "reason_code": gate.reason_code,
            "reason": gate.reason,
            "daily_loss": gate.daily_loss,
            "daily_loss_limit": gate.daily_loss_limit,
            "consecutive_losses": gate.consecutive_losses,
            "position_count": gate.position_count,
            "max_positions": gate.max_positions,
        }
    except Exception as e:
        logger.error("[claude_agent] get_risk_status エラー: %s", e, exc_info=True)
        return {"error": str(e)}
