"""
ユニットテスト: 8:50 の道の相関ゲートが、約定待ちの BUY も数える（#1320）

16:30 のルールの道（#1176）と Claude の道（#1177）と同じく、約定済みの保有と
約定待ちの BUY を合わせた銘柄で ENC を出す。売りシグナルと SL/TP の決済、
stats の n_held_symbols は約定済みの保有のまま。
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
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
_SL_PCT = 0.05
_GATE_OK = TradingGateStatus(is_allowed=True, stop_active=False, reason_code=None, reason=None)
_CORR_OK = CorrelationGateResult(
    is_allowed=True, enc=3.0, enc_threshold=2.0, avg_correlation=0.0, n_symbols=3
)


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
        "predicted_at": "20261028_083000",
        "current_price": current_price,
        "diff_ratio": diff_ratio,
        "confidence_ratio": 1.0,
    }


def _sl_check(avg_price, current_price):
    pnl = (current_price - avg_price) / avg_price
    triggered = pnl <= -_SL_PCT
    return triggered, False, "SL発動" if triggered else ""


def _same_moves(symbols, market, window):
    """9001 だけ相関 0 の値動きで、ほかは 9000 と同じ値動きのリターン"""
    days = np.arange(20)
    base = np.where(days % 2 == 0, 0.01, -0.01)
    other = np.where(days % 4 < 2, 0.01, -0.01)
    return pd.DataFrame({s: other if s == "9001" else base for s in symbols})


def _run(broker, predictions, corr_patch):
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
        patch("src.trading.execution.sl_tp.RiskManager", sl_risk),
        patch("src.trading.execution.RiskManager.evaluate_trading_gate", return_value=_GATE_OK),
        patch("src.trading.execution.RiskManager.calc_position_size", return_value=100),
        patch("src.trading.execution.RiskManager.update_peak_balance"),
        corr_patch,
        patch("src.trading.execution.runner._load_exit_model", return_value=None),
        patch("src.trading.execution.params.get_optimal_params", return_value={}),
        patch(
            "src.trading.execution.selection.get_symbol_sector",
            side_effect=lambda market, symbol: f"UNKNOWN:{market}:{symbol}",
        ),
    ]
    for p in patches:
        p.start()
    try:
        return run_daily_orders(
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


def _sent(broker):
    return [(c.args[0], c.args[1]) for c in broker.send_order.call_args_list]


class TestCorrelationGateCountsPendingBuys(unittest.TestCase):
    def test_gate_gets_held_and_pending(self):
        """保有1銘柄と約定待ちの BUY 2銘柄で、ゲートに3銘柄を渡す"""
        broker = _make_broker([{"symbol": "9000", "qty": 100, "avg_price": 1000.0}])
        broker.get_pending_buy_symbols = MagicMock(return_value={"P001", "P002.T"})
        gate = MagicMock(return_value=_CORR_OK)
        stats = _run(
            broker,
            pd.DataFrame([_row("7200", 0.03)]),
            patch("src.trading.execution.runner.evaluate_correlation_gate", gate),
        )
        self.assertEqual(gate.call_args.args[0], ["9000", "P001", "P002"])
        self.assertEqual(stats["n_held_symbols"], 1)
        self.assertEqual(stats["held_symbols_list"], ["9000"])

    def test_pending_same_moves_blocks_new_buy_but_exits_run(self):
        """約定待ちが保有と同じ値動きで ENC が線を下回れば新規は出さず、決済は出す

        保有の 9000 と 9001 は相関 0 で ENC=2.0（線ちょうどで通る）。約定待ちの P001・P002 が
        9000 と同じ値動きなので、4銘柄の ENC=4/(1+0.5*3)=1.6 で線を下回る。
        """
        positions = [
            {"symbol": "9000", "qty": 100, "avg_price": 1000.0, "current_price": 900.0},
            {"symbol": "9001", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0},
        ]
        broker = _make_broker(positions)
        broker.get_pending_buy_symbols = MagicMock(return_value={"P001", "P002"})
        predictions = pd.DataFrame([_row("9001", -0.03), _row("7200", 0.03)])
        with patch("src.trading.execution.runner.CORRELATION_ENC_THRESHOLD", 2.0):
            stats = _run(
                broker,
                predictions,
                patch("src.trading.correlation_risk._load_recent_returns", side_effect=_same_moves),
            )
        self.assertAlmostEqual(stats["enc"], 1.6)
        sent = _sent(broker)
        self.assertIn(("9000", OrderSide.SELL), sent)  # SL
        self.assertIn(("9001", OrderSide.SELL), sent)  # 売りシグナル
        self.assertNotIn(("7200", OrderSide.BUY), sent)
        self.assertEqual(stats["buy_orders"], 0)
        self.assertEqual(stats["sell_orders"], 2)
        self.assertTrue(stats["correlation_blocked"])

    def test_broker_without_reader_uses_held_only(self):
        """約定待ちを読む関数が無い Broker では、約定済みの保有だけで評価する"""
        broker = _make_broker([{"symbol": "9000", "qty": 100}, {"symbol": "9001", "qty": 100}])
        gate = MagicMock(return_value=_CORR_OK)
        _run(
            broker,
            pd.DataFrame([_row("7200", 0.03)]),
            patch("src.trading.execution.runner.evaluate_correlation_gate", gate),
        )
        self.assertEqual(gate.call_args.args[0], ["9000", "9001"])

    def test_reader_failure_uses_held_only(self):
        """約定待ちを読んで例外になる Broker では、約定済みの保有だけで評価する"""
        broker = _make_broker([{"symbol": "9000", "qty": 100}, {"symbol": "9001", "qty": 100}])
        broker.get_pending_buy_symbols = MagicMock(side_effect=RuntimeError("db down"))
        gate = MagicMock(return_value=_CORR_OK)
        stats = _run(
            broker,
            pd.DataFrame([_row("7200", 0.03)]),
            patch("src.trading.execution.runner.evaluate_correlation_gate", gate),
        )
        self.assertEqual(gate.call_args.args[0], ["9000", "9001"])
        self.assertEqual(stats["buy_orders"], 1)


if __name__ == "__main__":
    unittest.main()
