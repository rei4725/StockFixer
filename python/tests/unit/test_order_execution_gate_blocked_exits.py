"""
ユニットテスト: リスクのゲートが不合格の日も決済は走らせる（#874）

ゲートの不合格（日次損失・連続損失・保有の数の上限）で止めるのは新規の買いと
ショートの新規だけで、SL/TP の強制クローズと保有の売りシグナルの決済は出す。
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.infrastructure.in_memory import (
    InMemoryOrderRunSink,
    InMemoryPredictionRepository,
    InMemoryTradeDiffSink,
)
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.execution import run_daily_orders
from src.trading.types import CorrelationGateResult, TradingGateStatus

_PREDICTIONS = InMemoryPredictionRepository()

_BLOCKED_GATES = {
    "daily_loss_limit": TradingGateStatus(
        is_allowed=False,
        stop_active=True,
        reason_code="daily_loss_limit",
        reason="日次損失上限に到達: 損失=25000円 / 上限=20000円",
        daily_loss=25_000.0,
        daily_loss_limit=20_000.0,
    ),
    "consecutive_losses": TradingGateStatus(
        is_allowed=False,
        stop_active=True,
        reason_code="consecutive_losses",
        reason="連続損失 3 回 >= 3 回",
        daily_loss=0.0,
        daily_loss_limit=20_000.0,
    ),
    "max_positions": TradingGateStatus(
        is_allowed=False,
        stop_active=False,
        reason_code="max_positions",
        reason="保有銘柄数上限: 10 >= 10",
        daily_loss=0.0,
        daily_loss_limit=20_000.0,
    ),
}

_SL_PCT = 0.05


def _make_broker(positions):
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_balance.return_value = 1_000_000.0
    broker.get_positions.return_value = positions
    broker.get_short_positions = MagicMock(return_value=[])
    broker.send_order.return_value = {"order_id": "test-001", "status": "pending"}
    return broker


def _row(symbol, diff_ratio, current_price=1000.0):
    return {
        "market": "jp",
        "symbol": symbol,
        "predicted_at": "20261024_083000",
        "current_price": current_price,
        "diff_ratio": diff_ratio,
        "confidence_ratio": 1.0,
    }


def _sl_check(avg_price, current_price):
    pnl = (current_price - avg_price) / avg_price
    triggered = pnl <= -_SL_PCT
    return triggered, False, "SL発動" if triggered else ""


class TestGateBlockedStillExits(unittest.TestCase):
    def _run(self, gate_status, positions, predictions):
        sl_risk = MagicMock()
        sl_risk.return_value.check_sl_tp.side_effect = _sl_check
        patches = [
            patch.object(_PREDICTIONS, "get_latest_by_market", return_value=predictions),
            patch("src.trading.execution.runner._record_order"),
            patch("src.trading.execution.sl_tp._record_order"),
            patch(
                "src.trading.execution.runner._choose_order_params",
                return_value=(OrderType.MARKET, 0.0, "market", "open"),
            ),
            patch(
                "src.trading.execution.sl_tp._choose_order_params",
                return_value=(OrderType.MARKET, 0.0, "market", "open"),
            ),
            patch("src.trading.execution.sl_tp.RiskManager", sl_risk),
            patch(
                "src.trading.execution.RiskManager.evaluate_trading_gate",
                return_value=gate_status,
            ),
            patch("src.trading.execution.RiskManager.calc_position_size", return_value=100),
            patch("src.trading.execution.RiskManager.update_peak_balance"),
            patch(
                "src.trading.execution.runner.evaluate_correlation_gate",
                return_value=CorrelationGateResult(
                    is_allowed=True, enc=2.0, enc_threshold=2.0, avg_correlation=0.0, n_symbols=2
                ),
            ),
            patch("src.trading.execution.runner._load_exit_model", return_value=None),
            patch("src.trading.execution.params.get_optimal_params", return_value={}),
            patch(
                "src.trading.execution.selection.get_symbol_sector",
                side_effect=lambda market, symbol: f"UNKNOWN:{market}:{symbol}",
            ),
        ]
        for p in patches:
            p.start()
        broker = _make_broker(positions)
        try:
            stats = run_daily_orders(
                broker,
                order_run_sink=InMemoryOrderRunSink(),
                trade_diff_sink=InMemoryTradeDiffSink(),
                prediction_repo=_PREDICTIONS,
                market="jp",
                mode="paper",
            )
        finally:
            for p in patches:
                p.stop()
        return stats, broker

    @staticmethod
    def _sent(broker):
        return [(c.args[0], c.args[1]) for c in broker.send_order.call_args_list]

    def test_sl_exit_runs_when_gate_blocked(self):
        """ゲートの不合格の3つの理由のどれでも、SL の線を越えた保有は売る"""
        positions = [
            {"symbol": "9000", "qty": 100, "avg_price": 1000.0, "current_price": 900.0},
            {"symbol": "9001", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0},
        ]
        predictions = pd.DataFrame([_row("7200", 0.03)])
        for code, gate in _BLOCKED_GATES.items():
            with self.subTest(reason_code=code):
                stats, broker = self._run(gate, positions, predictions)
                sent = self._sent(broker)
                self.assertIn(("9000", OrderSide.SELL), sent)
                self.assertNotIn(("9001", OrderSide.SELL), sent)
                self.assertNotIn(("7200", OrderSide.BUY), sent)
                self.assertEqual(stats["sell_orders"], 1)
                self.assertEqual(stats["buy_orders"], 0)

    def test_sell_signal_exit_runs_and_new_buy_blocked(self):
        """ゲートの不合格でも保有の売りシグナルは売り、新規の買いは出さない"""
        positions = [
            {"symbol": "9000", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0},
        ]
        predictions = pd.DataFrame([_row("9000", -0.03), _row("7200", 0.03)])
        for code, gate in _BLOCKED_GATES.items():
            with self.subTest(reason_code=code):
                stats, broker = self._run(gate, positions, predictions)
                self.assertEqual(self._sent(broker), [("9000", OrderSide.SELL)])
                self.assertEqual(stats["sell_orders"], 1)
                self.assertEqual(stats["buy_orders"], 0)

    def test_stop_details_kept_when_gate_blocked(self):
        """stats の trading_stopped・stop_reason・reason_code は今のまま返る"""
        positions = [
            {"symbol": "9000", "qty": 100, "avg_price": 1000.0, "current_price": 900.0},
        ]
        predictions = pd.DataFrame([_row("7200", 0.03)])
        for code, gate in _BLOCKED_GATES.items():
            with self.subTest(reason_code=code):
                stats, _ = self._run(gate, positions, predictions)
                self.assertEqual(stats["trading_stopped"], gate.stop_active)
                self.assertEqual(stats["stop_reason"], gate.reason)
                self.assertEqual(stats["reason_code"], code)
                self.assertEqual(stats["daily_loss"], gate.daily_loss)
                self.assertEqual(stats["daily_loss_limit"], gate.daily_loss_limit)

    @patch("src.trading.execution.runner.ENABLE_SHORT_SIDE", True)
    def test_new_short_blocked_when_gate_blocked(self):
        """ゲートの不合格の日はショートの新規も出さない"""
        positions = [
            {"symbol": "9000", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0},
        ]
        predictions = pd.DataFrame([_row("8001", -0.03)])
        stats, broker = self._run(_BLOCKED_GATES["daily_loss_limit"], positions, predictions)
        self.assertEqual(stats["short_orders"], 0)
        broker.send_order.assert_not_called()


if __name__ == "__main__":
    unittest.main()
