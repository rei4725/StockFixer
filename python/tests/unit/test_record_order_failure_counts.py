"""ユニットテスト: send_order が通った後の付帯の記録の失敗（#1104）

send_order が order_id を返した後は注文が出たものとして数え（保有の上限・SL/TP の除外）、
_record_order の失敗は別に ERROR と errors に数える。ペーパーで期限を書けなかった
買い・空売りは取り消し、target_exit_date が NULL の pending を残さない。
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
from src.trading.brokers.paper.paper_broker import PaperBroker
from src.trading.execution import BUY_THRESHOLD, run_daily_orders
from src.trading.execution.recording import _record_order
from src.trading.types import TradingGateStatus
from src.utils.db._connection import _db_connection

_PREDICTIONS = InMemoryPredictionRepository()


def _broker(positions=None) -> MagicMock:
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_balance.return_value = 1_000_000.0
    broker.get_positions.return_value = positions or []
    broker.send_order.return_value = {"order_id": "ord-1104", "status": "pending"}
    return broker


def _row(symbol: str, diff_ratio: float) -> dict:
    return {
        "market": "jp",
        "symbol": symbol,
        "current_price": 800.0,
        "diff_ratio": diff_ratio,
        "confidence_ratio": 1.0,
    }


class TestRunnerCountsOrdersWhenRecordFails(unittest.TestCase):
    def setUp(self):
        sl_tp = patch("src.trading.execution.sl_tp.RiskManager")
        risk = sl_tp.start()
        self.addCleanup(sl_tp.stop)
        risk.return_value.check_sl_tp.return_value = (True, False, "SL")

    def _run(self, broker, predictions, extra_patches=()):
        gate = TradingGateStatus(
            is_allowed=True,
            stop_active=False,
            reason_code=None,
            reason=None,
            daily_loss=0.0,
            daily_loss_limit=None,
        )
        patches = [
            patch.object(_PREDICTIONS, "get_latest_by_market", return_value=predictions),
            patch(
                "src.trading.execution.runner._record_order",
                side_effect=RuntimeError("diff down"),
            ),
            patch(
                "src.trading.execution.sl_tp._record_order",
                side_effect=RuntimeError("diff down"),
            ),
            patch(
                "src.trading.execution.runner._choose_order_params",
                return_value=(OrderType.MARKET, 0.0, "market", "open"),
            ),
            patch(
                "src.trading.execution.RiskManager.evaluate_trading_gate",
                return_value=gate,
            ),
            patch("src.trading.execution.RiskManager.calc_position_size", return_value=100),
            patch(
                "src.trading.execution.selection.get_symbol_sector",
                side_effect=lambda market, symbol: f"UNKNOWN:{market}:{symbol}",
            ),
            *extra_patches,
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        with self.assertLogs("src.trading.execution", level="ERROR") as logs:
            stats = run_daily_orders(
                broker,
                order_run_sink=InMemoryOrderRunSink(),
                trade_diff_sink=InMemoryTradeDiffSink(),
                prediction_repo=_PREDICTIONS,
                market="jp",
                mode="paper",
            )
        return stats, logs

    def test_buy_does_not_exceed_max_positions(self):
        broker = _broker()
        predictions = pd.DataFrame(
            [_row(f"720{i}", BUY_THRESHOLD + 0.01 * (i + 1)) for i in range(3)]
        )

        stats, logs = self._run(
            broker,
            predictions,
            (patch("src.trading.execution.runner.MAX_POSITIONS", 1),),
        )

        buys = [c for c in broker.send_order.call_args_list if c.args[1] == OrderSide.BUY]
        self.assertEqual(len(buys), 1)
        self.assertEqual(stats["buy_orders"], 1)
        self.assertEqual(stats["errors"], 1)
        self.assertTrue(any("ord-1104" in line for line in logs.output))

    def test_sl_tp_symbol_gets_no_second_sell(self):
        held = [{"symbol": "9000", "qty": 100, "avg_price": 1000.0, "current_price": 800.0}]
        broker = _broker(positions=held)
        predictions = pd.DataFrame([_row("9000", -0.05)])

        stats, logs = self._run(broker, predictions)

        sells = [c for c in broker.send_order.call_args_list if c.args[1] == OrderSide.SELL]
        self.assertEqual(len(sells), 1)
        self.assertEqual(stats["sell_orders"], 1)
        self.assertEqual(stats["errors"], 1)
        self.assertTrue(any("ord-1104" in line for line in logs.output))


class TestPaperLinkFailureLeavesNoUndatedPending(unittest.TestCase):
    def setUp(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")
            con.execute("DELETE FROM paper_short_positions")
        self.broker = PaperBroker(
            market_data_port=MagicMock(), trade_diff_sink=InMemoryTradeDiffSink()
        )

    def _send_and_record(self, symbol: str, side: OrderSide) -> None:
        result = self.broker.send_order(symbol, side, 100, price=1000.0)
        with patch(
            "src.trading.execution.recording._link_paper_order_metadata",
            side_effect=RuntimeError("link down"),
        ):
            with self.assertRaises(RuntimeError):
                _record_order(
                    market="jp",
                    predicted_at="20261025_080000",
                    symbol=symbol,
                    side=side,
                    qty=100,
                    signal_price=1000.0,
                    order_price=1000.0,
                    order_type=OrderType.MARKET,
                    order_result=result,
                    broker=self.broker,
                    mode="paper",
                    horizon=5,
                    trade_diff_sink=InMemoryTradeDiffSink(),
                )

    def _undated_pending(self) -> list:
        with _db_connection() as con:
            return con.execute(
                "SELECT order_id FROM paper_orders "
                "WHERE status = 'pending' AND target_exit_date IS NULL"
            ).fetchall()

    def test_buy_is_cancelled(self):
        self._send_and_record("7203", OrderSide.BUY)

        self.assertEqual(self._undated_pending(), [])

    def test_short_is_cancelled_and_short_position_undone(self):
        with _db_connection() as con:
            con.execute(
                "INSERT INTO paper_short_positions "
                "(symbol, qty, avg_short_price, opened_at, updated_at) "
                "VALUES ('9984', 100, 900.0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )

        self._send_and_record("9984", OrderSide.SHORT)
        self._send_and_record("6758", OrderSide.SHORT)

        self.assertEqual(self._undated_pending(), [])
        with _db_connection() as con:
            rows = con.execute(
                "SELECT symbol, qty, avg_short_price FROM paper_short_positions ORDER BY symbol"
            ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0][0], int(rows[0][1])), ("9984", 100))
        self.assertAlmostEqual(float(rows[0][2]), 900.0)


if __name__ == "__main__":
    unittest.main()
