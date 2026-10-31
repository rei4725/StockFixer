"""
ユニットテスト: 8:50 の道の保有上限と業種の上限が、紙の約定待ちの BUY を数える（#1137）

前の回（16:30 のルールの道など）が出した BUY は 9:05 の約定まで get_positions に出ない。
数えないと、約定の後に MAX_POSITIONS と MAX_SECTOR_POSITIONS を超える。
"""

import unittest
from unittest.mock import MagicMock, patch

from tests.unit.test_order_execution_pipeline import _make_broker, _make_predictions

from config.settings import MAX_POSITIONS, MAX_SECTOR_POSITIONS
from src.domain.ports import MarketDataPort
from src.infrastructure.in_memory import (
    InMemoryOrderRunSink,
    InMemoryPredictionRepository,
    InMemoryTradeDiffSink,
)
from src.trading.brokers.base import OrderSide, OrderType
from src.trading.brokers.paper.paper_broker import PaperBroker
from src.trading.execution import run_daily_orders, selection
from src.trading.types import TradingGateStatus
from src.utils.db._connection import _db_connection

_PREDICTIONS = InMemoryPredictionRepository()


def _run(broker, predictions, sector_of):
    gate = TradingGateStatus(is_allowed=True, stop_active=False, reason_code=None, reason=None)
    with (
        patch.object(_PREDICTIONS, "get_latest_by_market", return_value=predictions),
        patch("src.trading.execution.runner._record_order"),
        patch(
            "src.trading.execution.runner._choose_order_params",
            return_value=(OrderType.MARKET, 0.0, "market", "open"),
        ),
        patch("src.trading.execution.RiskManager.evaluate_trading_gate", return_value=gate),
        patch("src.trading.execution.RiskManager.calc_position_size", return_value=100),
        patch("src.trading.risk_manager.RiskManager._get_daily_realized_loss", return_value=0.0),
        patch("src.trading.risk_manager.RiskManager._get_consecutive_losses", return_value=0),
        patch("src.trading.execution.selection.get_symbol_sector", side_effect=sector_of),
        patch("src.trading.execution.runner._load_pending_exits") as pending_exits,
    ):
        pending_exits.return_value.has.return_value = False
        return run_daily_orders(
            broker,
            order_run_sink=InMemoryOrderRunSink(),
            trade_diff_sink=InMemoryTradeDiffSink(),
            prediction_repo=_PREDICTIONS,
            market="jp",
            mode="paper",
        )


def _unique_sector(market: str, symbol: str) -> str:
    return f"UNKNOWN:{market}:{symbol}"


class TestPendingBuysInCaps(unittest.TestCase):
    def test_pending_buys_fill_position_cap_blocks_new_buys(self):
        """約定待ちの BUY が k 件・保有が MAX_POSITIONS − k 件なら新しく買わない。"""
        k = 2
        held = [{"symbol": f"S{i:03d}", "qty": 100} for i in range(MAX_POSITIONS - k)]
        broker = _make_broker(positions=held)
        broker.get_pending_buy_symbols = MagicMock(return_value={"P001", "P002"})
        stats = _run(broker, _make_predictions(n_buy=3), _unique_sector)
        self.assertEqual(stats["buy_orders"], 0)
        broker.send_order.assert_not_called()

    def test_pending_buy_sector_counts_toward_sector_cap(self):
        """約定待ちの BUY の業種が上限に達していれば、同じ業種の候補を落とす。"""
        pending = {f"P{i:03d}" for i in range(MAX_SECTOR_POSITIONS)}
        broker = _make_broker()
        broker.get_pending_buy_symbols = MagicMock(return_value=pending)

        def sector_of(market: str, symbol: str) -> str:
            return "Auto" if symbol.startswith("P") or symbol == "7200" else f"Other:{symbol}"

        stats = _run(broker, _make_predictions(n_buy=3), sector_of)
        bought = {c.args[0] for c in broker.send_order.call_args_list}
        self.assertNotIn("7200", bought)
        self.assertEqual(bought, {"7201", "7202"})
        self.assertEqual(stats["buy_orders"], 2)

    def test_pending_buy_symbol_not_bought_again(self):
        """約定待ちの BUY がある銘柄は候補から外す。"""
        broker = _make_broker()
        broker.get_pending_buy_symbols = MagicMock(return_value={"7200"})
        _run(broker, _make_predictions(n_buy=2), _unique_sector)
        bought = {c.args[0] for c in broker.send_order.call_args_list}
        self.assertEqual(bought, {"7201"})

    def test_broker_without_reader_counts_no_pending(self):
        """約定待ちを読む関数が無い Broker では空として数える。"""
        broker = _make_broker()
        self.assertEqual(selection._get_pending_buy_symbols(broker), set())

    def test_reader_failure_raises(self):
        """読めない日を0件と取り違えず、例外を上へ出す（#1465）。"""
        broker = _make_broker()
        broker.get_pending_buy_symbols = MagicMock(side_effect=RuntimeError("db down"))
        with self.assertRaises(RuntimeError):
            selection._get_pending_buy_symbols(broker)


class TestPaperBrokerPendingBuySymbols(unittest.TestCase):
    def setUp(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")
            con.execute("DELETE FROM paper_positions")
            con.execute("UPDATE paper_balance SET balance = 1000000.0")
        self.broker = PaperBroker(
            market_data_port=MagicMock(spec=MarketDataPort),
            trade_diff_sink=InMemoryTradeDiffSink(),
        )

    def test_returns_only_pending_buys(self):
        self.broker.send_order("7203", OrderSide.BUY, 100)
        self.broker.send_order("6758", OrderSide.BUY, 100)
        self.broker.send_order("9984", OrderSide.SELL, 100)
        cancelled = self.broker.send_order("8306", OrderSide.BUY, 100)
        self.broker.cancel_order(cancelled["order_id"])
        self.assertEqual(self.broker.get_pending_buy_symbols(), {"7203", "6758"})

    def test_empty_when_no_pending(self):
        self.assertEqual(self.broker.get_pending_buy_symbols(), set())
