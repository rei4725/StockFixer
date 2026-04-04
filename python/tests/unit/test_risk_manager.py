"""
ユニットテスト: RiskManager

外部依存（DuckDB・Broker）はすべて MagicMock で差し替える。
"""

import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from src.brokers.base import BrokerBase, OrderSide
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

    def test_daily_loss_status_contains_reason(self):
        balance = 1_000_000.0
        broker = _make_broker(balance=balance)
        loss = balance * MAX_DAILY_LOSS_RATE
        risk = self._make_risk(broker, daily_loss=loss)

        status = risk.evaluate_trading_gate()

        self.assertFalse(status.is_allowed)
        self.assertTrue(status.stop_active)
        self.assertEqual(status.reason_code, "daily_loss_limit")
        self.assertIn("日次損失上限", status.reason)
        self.assertEqual(status.daily_loss, loss)

    def test_daily_loss_guard_can_be_disabled_by_env(self):
        balance = 1_000_000.0
        broker = _make_broker(balance=balance)
        loss = balance * MAX_DAILY_LOSS_RATE * 3
        risk = self._make_risk(broker, daily_loss=loss)

        with patch.dict("os.environ", {"DISABLE_DAILY_LOSS_GUARD": "1"}, clear=False):
            status = risk.evaluate_trading_gate()

        self.assertTrue(status.is_allowed)
        self.assertIsNone(status.daily_loss_limit)
        risk._get_daily_realized_loss.assert_not_called()

    def test_daily_loss_rate_can_be_overridden_by_env(self):
        balance = 1_000_000.0
        broker = _make_broker(balance=balance)
        risk = self._make_risk(broker, daily_loss=15_000.0)

        with patch.dict("os.environ", {"MAX_DAILY_LOSS_RATE": "0.01"}, clear=False):
            status = risk.evaluate_trading_gate()

        self.assertFalse(status.is_allowed)
        self.assertEqual(status.daily_loss_limit, 10_000.0)

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

    def test_lower_confidence_ratio_reduces_position_size(self):
        risk = self._make_risk()
        qty_high_conf = risk.calc_position_size("7203", 1000.0, confidence_ratio=1.0)
        qty_low_conf = risk.calc_position_size("7203", 1000.0, confidence_ratio=0.4)

        self.assertGreaterEqual(qty_high_conf, qty_low_conf)
        self.assertGreater(qty_high_conf, 0)


class TestRiskManagerPersistence(unittest.TestCase):
    """DB参照を伴う内部ヘルパーのテスト"""

    def test_daily_realized_loss_uses_paper_orders(self):
        broker = _make_broker()
        risk = RiskManager(broker)

        mock_con = MagicMock()
        mock_con.execute.return_value.fetchone.return_value = (-1234.0,)

        @contextmanager
        def mock_db():
            yield mock_con

        with patch("src.services.risk_manager._db_connection", new=mock_db):
            daily_loss = risk._get_daily_realized_loss()

        self.assertEqual(daily_loss, 1234.0)
        query, params = mock_con.execute.call_args[0]
        self.assertIn("FROM paper_orders", query)
        self.assertEqual(params, [int(OrderSide.SELL)])

    def test_consecutive_losses_uses_paper_orders(self):
        broker = _make_broker()
        risk = RiskManager(broker)

        mock_con = MagicMock()
        mock_con.execute.return_value.fetchall.return_value = [(-100.0,), (-50.0,), (20.0,)]

        @contextmanager
        def mock_db():
            yield mock_con

        with patch("src.services.risk_manager._db_connection", new=mock_db):
            consecutive = risk._get_consecutive_losses()

        self.assertEqual(consecutive, 2)
        query, params = mock_con.execute.call_args[0]
        self.assertIn("FROM paper_orders", query)
        self.assertEqual(params, [int(OrderSide.SELL), MAX_CONSECUTIVE_LOSSES])

    def test_missing_trade_pnl_returns_zero_for_non_paper_broker(self):
        broker = _make_broker()
        broker.broker_name = "live"
        risk = RiskManager(broker)

        mock_con = MagicMock()
        mock_con.execute.return_value.fetchone.return_value = (0,)

        @contextmanager
        def mock_db():
            yield mock_con

        with patch("src.services.risk_manager._db_connection", new=mock_db):
            daily_loss = risk._get_daily_realized_loss()

        self.assertEqual(daily_loss, 0.0)
        query, params = mock_con.execute.call_args[0]
        self.assertIn("information_schema.tables", query)
        self.assertEqual(params, ["trade_pnl"])


if __name__ == "__main__":
    unittest.main()
