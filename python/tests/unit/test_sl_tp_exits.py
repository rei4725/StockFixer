"""
ユニットテスト: SL/TP 自動損切り・利確

- RiskManager.check_sl_tp の判定ロジック
- _check_sl_tp_exits の発注・stats 更新
"""

import unittest
from unittest.mock import MagicMock, patch

from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.execution import _check_sl_tp_exits
from src.trading.risk_manager import RiskManager


def _make_broker(positions=None):
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_balance.return_value = 1_000_000.0
    broker.get_positions.return_value = positions or []
    broker.send_order.return_value = {"order_id": "test-sl-001", "status": "pending"}
    return broker


def _make_stats():
    return {
        "buy_orders": 0,
        "sell_orders": 0,
        "short_orders": 0,
        "skipped": 0,
        "skipped_min_change": 0,
        "errors": 0,
        "trading_stopped": False,
        "stop_reason": None,
        "reason_code": None,
        "daily_loss": 0.0,
        "daily_loss_limit": None,
        "total_turnover": 0.0,
        "correlation_blocked": False,
        "enc": 0.0,
        "avg_correlation": 0.0,
        "n_held_symbols": 0,
        "held_symbols_list": [],
    }


class TestCheckSlTp(unittest.TestCase):
    def _make_risk(self, stop_loss_pct=None, take_profit_pct=None):
        broker = MagicMock(spec=BrokerBase)
        risk = RiskManager(broker)
        risk.stop_loss_pct = stop_loss_pct
        risk.take_profit_pct = take_profit_pct
        return risk

    def test_sl_triggered(self):
        risk = self._make_risk(stop_loss_pct=0.05)
        sl, tp, reason = risk.check_sl_tp(avg_price=1000.0, current_price=940.0)
        self.assertTrue(sl)
        self.assertFalse(tp)
        self.assertIn("SL発動", reason)

    def test_tp_triggered(self):
        risk = self._make_risk(take_profit_pct=0.10)
        sl, tp, reason = risk.check_sl_tp(avg_price=1000.0, current_price=1110.0)
        self.assertFalse(sl)
        self.assertTrue(tp)
        self.assertIn("TP発動", reason)

    def test_neither_triggered(self):
        risk = self._make_risk(stop_loss_pct=0.05, take_profit_pct=0.10)
        sl, tp, reason = risk.check_sl_tp(avg_price=1000.0, current_price=1020.0)
        self.assertFalse(sl)
        self.assertFalse(tp)
        self.assertEqual(reason, "")

    def test_no_thresholds(self):
        risk = self._make_risk()
        sl, tp, reason = risk.check_sl_tp(avg_price=1000.0, current_price=500.0)
        self.assertFalse(sl)
        self.assertFalse(tp)

    def test_avg_price_zero(self):
        risk = self._make_risk(stop_loss_pct=0.05)
        sl, tp, reason = risk.check_sl_tp(avg_price=0.0, current_price=1000.0)
        self.assertFalse(sl)
        self.assertFalse(tp)

    def test_sl_takes_priority_over_tp(self):
        # SL と TP の両方が設定されていても SL が優先される
        risk = self._make_risk(stop_loss_pct=0.01, take_profit_pct=0.01)
        # pnl_pct = -0.05 → SL ≥ 0.01 のみ発動
        sl, tp, reason = risk.check_sl_tp(avg_price=1000.0, current_price=950.0)
        self.assertTrue(sl)
        self.assertFalse(tp)

    def test_exact_sl_boundary(self):
        risk = self._make_risk(stop_loss_pct=0.05)
        # pnl_pct = -0.05 == -stop_loss_pct → 発動
        sl, tp, _ = risk.check_sl_tp(avg_price=1000.0, current_price=950.0)
        self.assertTrue(sl)

    def test_exact_tp_boundary(self):
        risk = self._make_risk(take_profit_pct=0.10)
        # pnl_pct = 0.10 == take_profit_pct → 発動
        sl, tp, _ = risk.check_sl_tp(avg_price=1000.0, current_price=1100.0)
        self.assertTrue(tp)


class TestCheckSlTpExits(unittest.TestCase):
    def _patch_risk(self, sl=False, tp=False, reason=""):
        return patch(
            "src.trading.execution.RiskManager.check_sl_tp",
            return_value=(sl, tp, reason),
        )

    def _make_market_data(self):
        return MagicMock()

    def test_sl_triggers_sell_order(self):
        positions = [{"symbol": "7203", "qty": 100, "avg_price": 1000.0, "current_price": 940.0}]
        broker = _make_broker(positions)
        stats = _make_stats()
        market_data = self._make_market_data()

        with self._patch_risk(sl=True, reason="SL発動: pnl=-6.00% <= -5.00%"):
            with patch("src.trading.execution.get_optimal_params", return_value={}):
                with patch(
                    "src.trading.execution._choose_order_params",
                    return_value=(OrderType.MARKET, 0.0, "market", "open"),
                ):
                    with patch("src.trading.execution._record_order"):
                        triggered = _check_sl_tp_exits(broker, "jp", "paper", market_data, stats)

        broker.send_order.assert_called_once_with(
            "7203", OrderSide.SELL, 100, price=0.0, order_type=OrderType.MARKET
        )
        self.assertIn("7203", triggered)
        self.assertEqual(stats["sell_orders"], 1)
        self.assertAlmostEqual(stats["total_turnover"], 940.0 * 100)

    def test_tp_triggers_sell_order(self):
        positions = [{"symbol": "9984", "qty": 200, "avg_price": 1000.0, "current_price": 1150.0}]
        broker = _make_broker(positions)
        stats = _make_stats()
        market_data = self._make_market_data()

        with self._patch_risk(tp=True, reason="TP発動: pnl=15.00% >= 10.00%"):
            with patch("src.trading.execution.get_optimal_params", return_value={}):
                with patch(
                    "src.trading.execution._choose_order_params",
                    return_value=(OrderType.MARKET, 0.0, "market", "open"),
                ):
                    with patch("src.trading.execution._record_order"):
                        triggered = _check_sl_tp_exits(broker, "jp", "paper", market_data, stats)

        self.assertIn("9984", triggered)
        self.assertEqual(stats["sell_orders"], 1)

    def test_no_trigger_no_order(self):
        positions = [{"symbol": "7203", "qty": 100, "avg_price": 1000.0, "current_price": 1010.0}]
        broker = _make_broker(positions)
        stats = _make_stats()
        market_data = self._make_market_data()

        with self._patch_risk(sl=False, tp=False, reason=""):
            with patch("src.trading.execution.get_optimal_params", return_value={}):
                triggered = _check_sl_tp_exits(broker, "jp", "paper", market_data, stats)

        broker.send_order.assert_not_called()
        self.assertEqual(len(triggered), 0)
        self.assertEqual(stats["sell_orders"], 0)

    def test_empty_positions_returns_empty_set(self):
        broker = _make_broker(positions=[])
        stats = _make_stats()
        market_data = self._make_market_data()

        triggered = _check_sl_tp_exits(broker, "jp", "paper", market_data, stats)
        self.assertEqual(triggered, set())
        broker.send_order.assert_not_called()

    def test_zero_qty_skipped(self):
        positions = [{"symbol": "7203", "qty": 0, "avg_price": 1000.0, "current_price": 900.0}]
        broker = _make_broker(positions)
        stats = _make_stats()
        market_data = self._make_market_data()

        with self._patch_risk(sl=True, reason="SL発動"):
            triggered = _check_sl_tp_exits(broker, "jp", "paper", market_data, stats)

        broker.send_order.assert_not_called()
        self.assertEqual(len(triggered), 0)

    def test_order_error_increments_errors(self):
        positions = [{"symbol": "7203", "qty": 100, "avg_price": 1000.0, "current_price": 900.0}]
        broker = _make_broker(positions)
        broker.send_order.side_effect = RuntimeError("send_order failed")
        stats = _make_stats()
        market_data = self._make_market_data()

        with self._patch_risk(sl=True, reason="SL発動"):
            with patch("src.trading.execution.get_optimal_params", return_value={}):
                with patch(
                    "src.trading.execution._choose_order_params",
                    return_value=(OrderType.MARKET, 0.0, "market", "open"),
                ):
                    triggered = _check_sl_tp_exits(broker, "jp", "paper", market_data, stats)

        self.assertEqual(stats["errors"], 1)
        self.assertEqual(len(triggered), 0)

    def test_symbol_dot_t_stripped(self):
        positions = [{"symbol": "7203.T", "qty": 100, "avg_price": 1000.0, "current_price": 900.0}]
        broker = _make_broker(positions)
        stats = _make_stats()
        market_data = self._make_market_data()

        with self._patch_risk(sl=True, reason="SL発動"):
            with patch("src.trading.execution.get_optimal_params", return_value={}):
                with patch(
                    "src.trading.execution._choose_order_params",
                    return_value=(OrderType.MARKET, 0.0, "market", "open"),
                ):
                    with patch("src.trading.execution._record_order"):
                        triggered = _check_sl_tp_exits(broker, "jp", "paper", market_data, stats)

        # send_order には .T 除去後の symbol が渡されること
        call_args = broker.send_order.call_args
        self.assertEqual(call_args[0][0], "7203")
        self.assertIn("7203", triggered)


if __name__ == "__main__":
    unittest.main()
