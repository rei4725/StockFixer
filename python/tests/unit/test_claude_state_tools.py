"""claude_state_tools の3つのハンドラーの戻りの形と例外の枝のテスト（#1856）。"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.trading.claude_state_tools import (
    _handle_get_balance,
    _handle_get_positions,
    _handle_get_risk_status,
)
from src.trading.types import TradingGateStatus

RISK_STATUS_KEYS = {
    "is_allowed",
    "stop_active",
    "reason_code",
    "reason",
    "daily_loss",
    "daily_loss_limit",
    "consecutive_losses",
    "position_count",
    "max_positions",
}


class TestHandleGetPositions:
    def test_returns_positions_from_broker(self) -> None:
        broker = MagicMock()
        positions = [{"symbol": "7203", "quantity": 100}]
        broker.get_positions.return_value = positions

        result = _handle_get_positions(broker)

        assert result == {"positions": positions}
        broker.get_positions.assert_called_once_with()

    def test_returns_error_when_broker_raises(self) -> None:
        broker = MagicMock()
        broker.get_positions.side_effect = RuntimeError("positions boom")

        result = _handle_get_positions(broker)

        assert result == {"error": "positions boom"}


class TestHandleGetBalance:
    def test_returns_balance_from_broker(self) -> None:
        broker = MagicMock()
        balance = {"cash": 1_000_000.0}
        broker.get_balance.return_value = balance

        result = _handle_get_balance(broker)

        assert result == {"balance": balance}
        broker.get_balance.assert_called_once_with()

    def test_returns_error_when_broker_raises(self) -> None:
        broker = MagicMock()
        broker.get_balance.side_effect = ValueError("balance boom")

        result = _handle_get_balance(broker)

        assert result == {"error": "balance boom"}


class TestHandleGetRiskStatus:
    def test_returns_nine_keys_matching_gate_values(self) -> None:
        gate = TradingGateStatus(
            is_allowed=False,
            stop_active=True,
            reason_code="daily_loss_limit",
            reason="日次損失上限に到達",
            daily_loss=25_000.0,
            daily_loss_limit=20_000.0,
            consecutive_losses=3,
            consecutive_loss_limit=5,
            position_count=4,
            max_positions=10,
        )
        risk = MagicMock()
        risk.evaluate_trading_gate.return_value = gate

        result = _handle_get_risk_status(risk)

        assert set(result.keys()) == RISK_STATUS_KEYS
        assert result == {
            "is_allowed": False,
            "stop_active": True,
            "reason_code": "daily_loss_limit",
            "reason": "日次損失上限に到達",
            "daily_loss": 25_000.0,
            "daily_loss_limit": 20_000.0,
            "consecutive_losses": 3,
            "position_count": 4,
            "max_positions": 10,
        }
        risk.evaluate_trading_gate.assert_called_once_with()

    def test_allowed_gate_with_defaults(self) -> None:
        gate = TradingGateStatus(is_allowed=True, stop_active=False)
        risk = MagicMock()
        risk.evaluate_trading_gate.return_value = gate

        result = _handle_get_risk_status(risk)

        assert set(result.keys()) == RISK_STATUS_KEYS
        for key in RISK_STATUS_KEYS:
            assert result[key] == getattr(gate, key)
        assert result["is_allowed"] is True
        assert result["reason_code"] is None
        assert result["daily_loss_limit"] is None
        assert result["max_positions"] is None

    def test_returns_error_when_gate_raises(self) -> None:
        risk = MagicMock()
        risk.evaluate_trading_gate.side_effect = RuntimeError("gate boom")

        result = _handle_get_risk_status(risk)

        assert result == {"error": "gate boom"}
