"""ユニットテスト: 空売りの保有も保有の上限（MAX_POSITIONS）に数える（#1146）

ゲートの保有銘柄数と 8:50 の発注の中の数えが買い持ちだけを見ていたため、空売りを
有効にすると、建玉の合計が MAX_POSITIONS を超えるまで空売りの新規を建てられた。
約定待ちの BUY の銘柄も新規ショートの候補から外す（#1333）。
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
from src.trading.risk_manager import MAX_POSITIONS, RiskManager
from src.trading.types import TradingGateStatus
from src.utils.db._connection import _db_connection

_PREDICTIONS = InMemoryPredictionRepository()


def _broker(positions=None, short_positions=None) -> MagicMock:
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_balance.return_value = 1_000_000.0
    broker.get_positions.return_value = positions or []
    broker.get_short_positions.return_value = short_positions or []
    broker.send_order.return_value = {"order_id": "ord-1146", "status": "pending"}
    return broker


def _long(symbol: str) -> dict:
    return {"symbol": symbol, "qty": 100, "avg_price": 1000.0, "current_price": 1000.0}


def _short(symbol: str) -> dict:
    return {"symbol": symbol, "qty": 100, "avg_short_price": 1000.0, "current_price": 1000.0}


def _row(symbol: str, diff_ratio: float) -> dict:
    return {
        "market": "jp",
        "symbol": symbol,
        "current_price": 800.0,
        "diff_ratio": diff_ratio,
        "confidence_ratio": 1.0,
    }


class TestGateCountsShortPositions(unittest.TestCase):
    def _gate(self, broker) -> TradingGateStatus:
        risk = RiskManager(broker)
        risk._get_daily_realized_loss = MagicMock(return_value=0.0)
        risk._get_consecutive_losses = MagicMock(return_value=0)
        return risk.evaluate_trading_gate()

    def test_long_plus_short_at_limit_blocks_new_entries(self):
        longs = [_long(f"10{i:02d}") for i in range(MAX_POSITIONS - 1)]
        broker = _broker(positions=longs, short_positions=[_short("9984")])

        status = self._gate(broker)

        self.assertFalse(status.is_allowed)
        self.assertEqual(status.reason_code, "max_positions")
        self.assertEqual(status.position_count, MAX_POSITIONS)

    def test_unreadable_short_positions_raise_instead_of_counting_longs_only(self):
        """空売りが読めない broker では買い持ちだけで数えず例外を出す（#1458）"""
        broker = MagicMock(spec=["get_balance", "get_positions", "broker_name"])
        broker.broker_name = "paper"
        broker.get_balance.return_value = 1_000_000.0
        broker.get_positions.return_value = [_long("7203")]

        with self.assertRaises(AttributeError):
            self._gate(broker)


class TestRunnerCountsShortPositions(unittest.TestCase):
    def setUp(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")

    def _run(self, broker, predictions, max_positions: int):
        gate = TradingGateStatus(
            is_allowed=True,
            stop_active=False,
            reason_code=None,
            reason=None,
            daily_loss=0.0,
            daily_loss_limit=None,
        )
        corr = MagicMock(is_allowed=True, enc=0.0, avg_correlation=0.0, reason=None)
        patches = [
            patch.object(_PREDICTIONS, "get_latest_by_market", return_value=predictions),
            patch("src.trading.execution.runner._record_order"),
            patch(
                "src.trading.execution.runner._choose_order_params",
                return_value=(OrderType.MARKET, 0.0, "market", "open"),
            ),
            patch("src.trading.execution.RiskManager.evaluate_trading_gate", return_value=gate),
            patch("src.trading.execution.RiskManager.calc_position_size", return_value=100),
            patch("src.trading.execution.sl_tp.RiskManager"),
            patch("src.trading.execution.runner.evaluate_correlation_gate", return_value=corr),
            patch("src.trading.execution.runner.ENABLE_SHORT_SIDE", True),
            patch("src.trading.execution.runner.MAX_POSITIONS", max_positions),
            patch(
                "src.trading.execution.selection.get_symbol_sector",
                side_effect=lambda market, symbol: f"UNKNOWN:{market}:{symbol}",
            ),
        ]
        for p in patches:
            started = p.start()
            self.addCleanup(p.stop)
            if p.attribute == "RiskManager":
                started.return_value.check_sl_tp.return_value = (False, False, "")
        return run_daily_orders(
            broker,
            order_run_sink=InMemoryOrderRunSink(),
            trade_diff_sink=InMemoryTradeDiffSink(),
            prediction_repo=_PREDICTIONS,
            market="jp",
            mode="paper",
        )

    @staticmethod
    def _shorts_sent(broker: MagicMock) -> list[str]:
        return [c.args[0] for c in broker.send_order.call_args_list if c.args[1] == OrderSide.SHORT]

    def test_no_short_entry_when_long_plus_short_at_limit(self):
        broker = _broker(positions=[_long("7203"), _long("6758")], short_positions=[_short("9984")])
        predictions = pd.DataFrame([_row("8306", -0.05), _row("4063", -0.05)])

        stats = self._run(broker, predictions, max_positions=3)

        self.assertEqual(self._shorts_sent(broker), [])
        self.assertEqual(stats["short_orders"], 0)

    def test_short_entries_stop_when_run_reaches_limit(self):
        broker = _broker(positions=[_long("7203"), _long("6758")])
        predictions = pd.DataFrame([_row("8306", -0.05), _row("4063", -0.05)])

        stats = self._run(broker, predictions, max_positions=3)

        self.assertEqual(len(self._shorts_sent(broker)), 1)
        self.assertEqual(stats["short_orders"], 1)

    def test_no_short_entry_on_pending_buy_symbol(self):
        """約定待ちの BUY がある銘柄には新規ショートを出さない（#1333）。"""
        broker = _broker()
        broker.get_pending_buy_symbols = MagicMock(return_value={"8306"})
        predictions = pd.DataFrame([_row("8306", -0.05), _row("4063", -0.05)])

        self._run(broker, predictions, max_positions=10)

        self.assertEqual(self._shorts_sent(broker), ["4063"])


if __name__ == "__main__":
    unittest.main()
