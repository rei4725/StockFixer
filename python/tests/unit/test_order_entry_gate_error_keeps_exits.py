"""ユニットテスト: 新規にだけ使う手順が落ちても、売りシグナルの決済は出す（#1281）

DD の基準の更新・リスクのゲート・相関ゲートのどれかが例外になった日も、予測と保有が
読めていれば、新規（BUY・SHORT）は止めたまま、売りシグナルの SELL と空売りの
SHORT_COVER を出してから、同じ例外を投げ直す（16:30 の道の #1221 と同じ形）。
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

_GATE = TradingGateStatus(
    is_allowed=True,
    stop_active=False,
    reason_code=None,
    reason=None,
    daily_loss=0.0,
    daily_loss_limit=None,
)
_CORR = CorrelationGateResult(
    is_allowed=True, enc=5.0, enc_threshold=2.0, avg_correlation=0.1, n_symbols=1
)


def _row(symbol: str, diff_ratio: float) -> dict:
    return {
        "market": "jp",
        "symbol": symbol,
        "current_price": 1000.0,
        "diff_ratio": diff_ratio,
        "confidence_ratio": 1.0,
    }


# 9001 は保有で売りシグナル、6758 は空売りで買いシグナル、
# 7000 は新規の買い、6000 は新規のショートの候補
_PREDICTIONS = pd.DataFrame(
    [_row("9001", -0.05), _row("6758", 0.05), _row("7000", 0.05), _row("6000", -0.05)]
)


def _broker() -> MagicMock:
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_balance.return_value = 1_000_000.0
    broker.get_positions.return_value = [
        {"symbol": "9001", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0}
    ]
    broker.get_short_positions.return_value = [
        {"symbol": "6758", "qty": 100, "avg_short_price": 1000.0, "current_price": 1000.0}
    ]
    broker.send_order.return_value = {"order_id": "ord-1281", "status": "pending"}
    return broker


class TestEntryGateErrorKeepsExits(unittest.TestCase):
    def setUp(self):
        self.broker = _broker()
        self.repo = InMemoryPredictionRepository()
        patches = {
            "repo": patch.object(
                self.repo, "get_latest_by_market", return_value=_PREDICTIONS.copy()
            ),
            "record": patch("src.trading.execution.runner._record_order"),
            "params": patch(
                "src.trading.execution.runner._choose_order_params",
                return_value=(OrderType.MARKET, 0.0, "market", "open"),
            ),
            "peak": patch("src.trading.execution.RiskManager.update_peak_balance"),
            "gate": patch(
                "src.trading.execution.RiskManager.evaluate_trading_gate", return_value=_GATE
            ),
            "size": patch("src.trading.execution.RiskManager.calc_position_size", return_value=100),
            "corr": patch(
                "src.trading.execution.runner.evaluate_correlation_gate", return_value=_CORR
            ),
            "sl_tp": patch("src.trading.execution.sl_tp.RiskManager"),
            "sector": patch(
                "src.trading.execution.selection.get_symbol_sector",
                side_effect=lambda market, symbol: f"UNKNOWN:{market}:{symbol}",
            ),
            "short": patch("src.trading.execution.runner.ENABLE_SHORT_SIDE", True),
            "pending": patch("src.trading.execution.runner._load_pending_exits"),
            "pending_buy": patch(
                "src.trading.execution.runner._get_pending_buy_symbols", return_value=set()
            ),
            "exit_model": patch("src.trading.execution.runner._load_exit_model", return_value=None),
        }
        self.mocks = {}
        for name, p in patches.items():
            self.mocks[name] = p.start()
            self.addCleanup(p.stop)
        self.mocks["sl_tp"].return_value.check_sl_tp.return_value = (False, False, "")
        self.mocks["pending"].return_value.has.return_value = False

    def _run(self):
        return run_daily_orders(
            self.broker,
            order_run_sink=InMemoryOrderRunSink(),
            trade_diff_sink=InMemoryTradeDiffSink(),
            prediction_repo=self.repo,
            market="jp",
            mode="paper",
        )

    def _sent(self) -> list[tuple[str, OrderSide]]:
        return [(c.args[0], c.args[1]) for c in self.broker.send_order.call_args_list]

    def _assert_exits_only_then_raise(self, error: Exception):
        with self.assertRaises(RuntimeError) as ctx:
            self._run()
        self.assertIs(ctx.exception, error)
        self.assertEqual(self._sent(), [("9001", OrderSide.SELL), ("6758", OrderSide.SHORT_COVER)])

    def test_no_error_sends_exits_and_new_entries(self):
        self._run()
        self.assertEqual(
            self._sent(),
            [
                ("9001", OrderSide.SELL),
                ("7000", OrderSide.BUY),
                ("6758", OrderSide.SHORT_COVER),
                ("6000", OrderSide.SHORT),
            ],
        )

    def test_update_peak_balance_error(self):
        error = RuntimeError("dd_state が読めない")
        self.mocks["peak"].side_effect = error
        self._assert_exits_only_then_raise(error)

    def test_evaluate_trading_gate_error(self):
        error = RuntimeError("日次損失の SQL が落ちた")
        self.mocks["gate"].side_effect = error
        self._assert_exits_only_then_raise(error)

    def test_evaluate_correlation_gate_error(self):
        error = RuntimeError("pivot failed")
        self.mocks["corr"].side_effect = error
        self._assert_exits_only_then_raise(error)


if __name__ == "__main__":
    unittest.main()
