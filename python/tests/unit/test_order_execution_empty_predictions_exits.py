"""
ユニットテスト: 予測が空の日も SL/TP の強制決済は走らせる（#1049）

SL/TP の判定は予測を使わないので、予測の道が落ちた日（データの止まり・モデルの欠け）も
SL の線を割った保有は売る。予測を使う新規の買い・ショートと売りシグナルの決済は出さない。
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
from src.trading.types import TradingGateStatus

_PREDICTIONS = InMemoryPredictionRepository()

_ALLOWED_GATE = TradingGateStatus(
    is_allowed=True,
    stop_active=False,
    reason_code=None,
    reason=None,
    daily_loss=1_000.0,
    daily_loss_limit=20_000.0,
)

_SL_PCT = 0.05


def _make_broker(positions):
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_balance.return_value = 1_000_000.0
    broker.get_positions.return_value = positions
    broker.get_short_positions = MagicMock(return_value=[])
    broker.send_order.return_value = {"order_id": "test-001", "status": "pending"}
    return broker


def _sl_check(avg_price, current_price):
    pnl = (current_price - avg_price) / avg_price
    triggered = pnl <= -_SL_PCT
    return triggered, False, "SL発動" if triggered else ""


class TestEmptyPredictionsStillExits(unittest.TestCase):
    def _run(self, positions):
        sl_risk = MagicMock()
        sl_risk.return_value.check_sl_tp.side_effect = _sl_check
        patches = [
            patch.object(_PREDICTIONS, "get_latest_by_market", return_value=pd.DataFrame()),
            patch("src.trading.execution.runner._record_order"),
            patch("src.trading.execution.sl_tp._record_order"),
            patch(
                "src.trading.execution.sl_tp._choose_order_params",
                return_value=(OrderType.MARKET, 0.0, "market", "open"),
            ),
            patch("src.trading.execution.sl_tp.RiskManager", sl_risk),
            patch(
                "src.trading.execution.RiskManager.evaluate_trading_gate",
                return_value=_ALLOWED_GATE,
            ),
            patch("src.trading.execution.RiskManager.calc_position_size", return_value=100),
            patch("src.trading.execution.RiskManager.update_peak_balance"),
            patch("src.trading.execution.runner.ENABLE_SHORT_SIDE", True),
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

    def test_sl_exit_runs_when_predictions_empty(self):
        """予測が空でも、SL の線を割った保有は売り、割っていない保有は売らない"""
        positions = [
            {"symbol": "9000", "qty": 100, "avg_price": 1000.0, "current_price": 900.0},
            {"symbol": "9001", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0},
        ]
        stats, broker = self._run(positions)
        self.assertEqual(self._sent(broker), [("9000", OrderSide.SELL)])
        self.assertEqual(stats["sell_orders"], 1)

    def test_no_new_buy_or_short_when_predictions_empty(self):
        """予測が空の日は新規の買いとショートを出さない"""
        positions = [
            {"symbol": "9001", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0},
        ]
        stats, broker = self._run(positions)
        broker.send_order.assert_not_called()
        self.assertEqual(stats["buy_orders"], 0)
        self.assertEqual(stats["short_orders"], 0)

    def test_daily_loss_kept_when_predictions_empty(self):
        """予測が空の日も daily_loss・daily_loss_limit は今のまま返る"""
        stats, _ = self._run([])
        self.assertEqual(stats["daily_loss"], _ALLOWED_GATE.daily_loss)
        self.assertEqual(stats["daily_loss_limit"], _ALLOWED_GATE.daily_loss_limit)


if __name__ == "__main__":
    unittest.main()
