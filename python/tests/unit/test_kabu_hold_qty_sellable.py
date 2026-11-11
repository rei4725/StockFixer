"""ユニットテスト: 返済の注文で拘束された数量（HoldQty）を売りに出さない（#1764）

kabu の /positions の LeavesQty（残数量）は、注文中に拘束された HoldQty を含む。
SL/TP と予想の売りの売る数は LeavesQty − HoldQty にし、0 なら売りを出さない。
qty（保有の上限・総資産の時価が使う）は LeavesQty のままにする。
hold_qty の欄が無い保有（紙のブローカー）は今までどおり qty を売る。
"""

import logging
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd

from src.infrastructure.in_memory import (
    InMemoryOrderRunSink,
    InMemoryPredictionRepository,
    InMemoryTradeDiffSink,
)
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.execution import _check_sl_tp_exits, run_daily_orders
from src.trading.position_valuation import positions_market_value, sellable_long_qty
from src.trading.types import TradingGateStatus
from src.utils.db._connection import _db_connection

_PREDICTIONS = InMemoryPredictionRepository()
_GET = "src.trading.brokers.kabu.kabu_client.httpx.get"


def _kabu_positions(rows):
    from src.trading.brokers.kabu.kabu_client import KabuBroker

    broker = KabuBroker(api_password="test_password")
    broker._token = "cached_token"
    broker._token_expires_at = datetime.now() + timedelta(hours=1)
    resp = MagicMock()
    resp.json.return_value = rows
    resp.raise_for_status = MagicMock()
    with patch(_GET, return_value=resp):
        return broker.get_positions()


def _broker(positions) -> MagicMock:
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_balance.return_value = 1_000_000.0
    broker.get_positions.return_value = positions
    broker.get_short_positions.return_value = []
    broker.send_order.return_value = {"order_id": "ord-1764", "status": "pending"}
    return broker


def _stats() -> dict:
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


def _long(symbol: str, hold_qty: int | None = None) -> dict:
    pos = {"symbol": symbol, "qty": 100, "avg_price": 1000.0, "current_price": 800.0}
    if hold_qty is not None:
        pos["hold_qty"] = hold_qty
    return pos


def _row(symbol: str) -> dict:
    return {
        "market": "jp",
        "symbol": symbol,
        "current_price": 800.0,
        "diff_ratio": -0.05,
        "confidence_ratio": 1.0,
    }


def _sells(broker: MagicMock) -> dict[str, int]:
    return {
        c.args[0]: c.args[2]
        for c in broker.send_order.call_args_list
        if c.args[1] == OrderSide.SELL
    }


class TestKabuPositionRowHoldQty(unittest.TestCase):
    def test_hold_qty_is_kept_apart_from_qty(self):
        pos = _kabu_positions(
            [
                {
                    "Symbol": "7203",
                    "LeavesQty": 300,
                    "HoldQty": 100,
                    "Price": 1000.0,
                    "CurrentPrice": 1100.0,
                }
            ]
        )[0]
        self.assertEqual(pos["qty"], 300)
        self.assertEqual(pos["hold_qty"], 100)
        self.assertEqual(sellable_long_qty(pos), 200)
        value = positions_market_value([pos], logging.getLogger(__name__))
        self.assertEqual(value, 300 * 1100.0)

    def test_null_hold_qty_becomes_zero(self):
        pos = _kabu_positions(
            [
                {
                    "Symbol": "7203",
                    "LeavesQty": 300,
                    "HoldQty": None,
                    "Price": 1000.0,
                    "CurrentPrice": 1100.0,
                }
            ]
        )[0]
        self.assertEqual(pos["hold_qty"], 0)
        self.assertEqual(sellable_long_qty(pos), 300)

    def test_without_hold_qty_sells_qty(self):
        self.assertEqual(sellable_long_qty(_long("7203")), 100)

    def test_hold_qty_over_qty_is_not_negative(self):
        self.assertEqual(sellable_long_qty(_long("7203", hold_qty=150)), 0)


class _DbCase(unittest.TestCase):
    def setUp(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")
        self.addCleanup(self._clear)

    def _clear(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")


class TestSlTpSellsUnheldQty(_DbCase):
    def _run(self, broker):
        with patch("src.trading.execution.sl_tp.RiskManager") as risk:
            risk.return_value.check_sl_tp.return_value = (True, False, "SL")
            with patch("src.trading.execution.sl_tp._record_order"):
                return _check_sl_tp_exits(
                    broker, "jp", "paper", MagicMock(), _stats(), InMemoryTradeDiffSink()
                )

    def test_sl_tp_sells_leaves_minus_hold(self):
        broker = _broker([_long("7203", hold_qty=40), _long("9000", hold_qty=100), _long("6758")])

        triggered = self._run(broker)

        self.assertEqual(_sells(broker), {"7203": 60, "6758": 100})
        self.assertNotIn("9000", triggered)


class TestRunnerSellsUnheldQty(_DbCase):
    def _run(self, broker, predictions):
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
            patch("src.trading.execution.runner._record_order"),
            patch(
                "src.trading.execution.runner._choose_order_params",
                return_value=(OrderType.MARKET, 0.0, "market", "open"),
            ),
            patch(
                "src.trading.execution.RiskManager.evaluate_trading_gate",
                return_value=gate,
            ),
            patch("src.trading.execution.RiskManager.calc_position_size", return_value=0),
            patch("src.trading.execution.sl_tp.RiskManager"),
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

    def test_sell_signal_sells_leaves_minus_hold(self):
        broker = _broker([_long("7203", hold_qty=40), _long("9000", hold_qty=100), _long("6758")])
        predictions = pd.DataFrame([_row("7203"), _row("9000"), _row("6758")])

        stats = self._run(broker, predictions)

        self.assertEqual(_sells(broker), {"7203": 60, "6758": 100})
        self.assertEqual(stats["sell_orders"], 2)


if __name__ == "__main__":
    unittest.main()
