"""
ユニットテスト: RiskManager

外部依存（DuckDB・Broker）はすべて MagicMock で差し替える。
"""

import unittest
from unittest.mock import MagicMock

from src.brokers.base import BrokerBase
from src.services.risk_manager import (
    MAX_CONSECUTIVE_LOSSES,
    MAX_DAILY_LOSS_RATE,
    MAX_POSITION_RATE,
    MAX_POSITIONS,
    RiskManager,
)


def _make_broker(balance: float = 1_000_000.0, positions: list | None = None) -> MagicMock:
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_balance.return_value = balance
    broker.get_positions.return_value = positions or []
    return broker


class TestRiskManagerIsAllowed(unittest.TestCase):
    """is_trading_allowed() のゲートチェックテスト"""

    def _make_risk(self, broker, daily_loss=0.0, consecutive=0):
        risk = RiskManager(broker)
        risk._get_daily_realized_loss = MagicMock(return_value=daily_loss)
        risk._get_consecutive_losses = MagicMock(return_value=consecutive)
        return risk

    def test_all_clear_returns_true(self):
        broker = _make_broker()
        risk = self._make_risk(broker)
        self.assertTrue(risk.is_trading_allowed())

    def test_daily_loss_exceed_blocks(self):
        balance = 1_000_000.0
        broker = _make_broker(balance=balance)
        # 損失が残高の2%ちょうど → ブロック
        loss = balance * MAX_DAILY_LOSS_RATE
        risk = self._make_risk(broker, daily_loss=loss)
        self.assertFalse(risk.is_trading_allowed())

    def test_consecutive_loss_blocks(self):
        broker = _make_broker()
        risk = self._make_risk(broker, consecutive=MAX_CONSECUTIVE_LOSSES)
        self.assertFalse(risk.is_trading_allowed())

    def test_max_positions_blocks_new_buy(self):
        positions = [{"symbol": f"100{i}", "qty": 100} for i in range(MAX_POSITIONS)]
        broker = _make_broker(positions=positions)
        risk = self._make_risk(broker)
        # 保有銘柄数が上限に達しているので False
        self.assertFalse(risk.is_trading_allowed())

    def test_under_limit_returns_true(self):
        positions = [{"symbol": "7203", "qty": 100}]
        broker = _make_broker(positions=positions)
        risk = self._make_risk(broker)
        self.assertTrue(risk.is_trading_allowed())


class TestRiskManagerCalcPositionSize(unittest.TestCase):
    """calc_position_size() のポジションサイジングテスト"""

    def _make_risk(self, balance=1_000_000.0):
        broker = _make_broker(balance=balance)
        risk = RiskManager(broker)
        return risk

    def test_returns_multiple_of_100(self):
        risk = self._make_risk()
        qty = risk.calc_position_size("7203", price=1000.0)
        self.assertEqual(qty % 100, 0)

    def test_zero_price_returns_zero(self):
        risk = self._make_risk()
        qty = risk.calc_position_size("7203", price=0.0)
        self.assertEqual(qty, 0)

    def test_within_position_cap(self):
        balance = 1_000_000.0
        price = 500.0
        risk = self._make_risk(balance=balance)
        qty = risk.calc_position_size("7203", price=price)
        # 上限: balance × 10% / price
        self.assertLessEqual(qty * price, balance * MAX_POSITION_RATE + price)

    def test_high_win_rate_yields_more_than_low(self):
        risk = self._make_risk()
        qty_high = risk.calc_position_size(
            "7203", 1000.0, win_rate=0.7, avg_win=0.02, avg_loss=0.01
        )
        qty_low = risk.calc_position_size("7203", 1000.0, win_rate=0.4, avg_win=0.02, avg_loss=0.01)
        self.assertGreaterEqual(qty_high, qty_low)


if __name__ == "__main__":
    unittest.main()
