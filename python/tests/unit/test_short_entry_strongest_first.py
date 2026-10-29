"""ユニットテスト: 新規ショートは下げの見込みの大きい順に上限まで取る（#1383）

予想は diff_ratio の大きい順に並んで渡る。並べ直さずに head を取っていたため、候補が
MAX_ORDERS_PER_RUN より多い日は、売りの線に近い弱い銘柄から空売りを建てていた。
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
from src.trading.execution import MAX_ORDERS_PER_RUN, run_daily_orders
from src.trading.types import TradingGateStatus
from src.utils.db._connection import _db_connection

_PREDICTIONS = InMemoryPredictionRepository()


def _broker() -> MagicMock:
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_balance.return_value = 1_000_000.0
    broker.get_positions.return_value = []
    broker.get_short_positions.return_value = []
    broker.send_order.return_value = {"order_id": "ord-1383", "status": "pending"}
    return broker


def _predictions() -> pd.DataFrame:
    """売りの線より下の候補を MAX_ORDERS_PER_RUN + 2 件、diff_ratio の大きい順で返す。"""
    n = MAX_ORDERS_PER_RUN + 2
    rows = [
        {
            "market": "jp",
            "symbol": f"{8000 + i}",
            "current_price": 800.0,
            "diff_ratio": -0.02 - 0.01 * i,
            "confidence_ratio": 1.0,
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


class TestShortEntryStrongestFirst(unittest.TestCase):
    def setUp(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")

    def _run(self, broker, predictions, *, gate_allowed: bool = True):
        gate = TradingGateStatus(
            is_allowed=gate_allowed,
            stop_active=False,
            reason_code=None if gate_allowed else "daily_loss",
            reason=None if gate_allowed else "日次損失上限",
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
            patch("src.trading.execution.runner.MAX_POSITIONS", 100),
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

    def test_short_entries_take_lowest_scores_up_to_limit(self):
        broker = _broker()
        predictions = _predictions()
        expected = set(predictions["symbol"].iloc[-MAX_ORDERS_PER_RUN:])

        stats = self._run(broker, predictions)

        self.assertEqual(set(self._shorts_sent(broker)), expected)
        self.assertEqual(stats["short_orders"], MAX_ORDERS_PER_RUN)

    def test_no_short_entry_when_gate_blocked(self):
        broker = _broker()

        stats = self._run(broker, _predictions(), gate_allowed=False)

        self.assertEqual(self._shorts_sent(broker), [])
        self.assertEqual(stats["short_orders"], 0)


if __name__ == "__main__":
    unittest.main()
