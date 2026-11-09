"""KabuBroker の約定待ちの読み取りと、8:50 の道の重ね買いの止め (#1521)。

kabu の /orders の応答の State は 1〜4 が終わっていない注文、5 が終了（全部約定・
取消・失効・発注エラー）。Side は "2" が買、"1" が売。
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

if "httpx" not in sys.modules:
    sys.modules["httpx"] = MagicMock()

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from tests.unit.test_order_corr_gate_pending import _CORR_OK, _row  # noqa: E402
from tests.unit.test_order_corr_gate_pending import _run as _run_with_exits  # noqa: E402
from tests.unit.test_order_execution_pipeline import _make_predictions  # noqa: E402
from tests.unit.test_pending_buys_in_caps import _run, _unique_sector  # noqa: E402

from config.settings import MAX_POSITIONS  # noqa: E402
from src.trading.brokers.base import OrderSide  # noqa: E402
from src.trading.brokers.kabu.kabu_client import KabuBroker  # noqa: E402
from src.trading.execution.pending import _load_pending_exits  # noqa: E402

_GET = "src.trading.brokers.kabu.kabu_client.httpx.get"


def _order(symbol, side, state, order_id="ORD-1"):
    return {
        "ID": order_id,
        "Symbol": symbol,
        "Side": side,
        "OrderQty": 100,
        "Price": 0.0,
        "State": state,
        "CumQty": 0,
        "Details": [],
    }


_ORDERS = [
    _order("7203", "2", 3, "ORD-WAIT"),  # 発注済で約定待ち
    _order("6758", "2", 1, "ORD-QUEUED"),  # 発注待機
    _order("9984", "2", 5, "ORD-FILLED"),  # 全部約定で終了
    _order("8306", "2", 5, "ORD-CANCEL"),  # 取消で終了
    _order("4063", "1", 3, "ORD-SELL"),  # 約定待ちの売り
]


def _resp(raw):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = raw
    return resp


def _kabu_broker(positions=None):
    broker = KabuBroker(api_password="pw")
    broker.get_token = MagicMock(return_value="tok")  # type: ignore[method-assign]
    broker.get_positions = MagicMock(return_value=positions or [])  # type: ignore[method-assign]
    broker.get_balance = MagicMock(return_value=1_000_000.0)  # type: ignore[method-assign]
    broker.send_order = MagicMock(  # type: ignore[method-assign]
        return_value={"order_id": "ORD-NEW", "status": "accepted"}
    )
    return broker


class TestKabuPendingSymbols:
    def test_pending_buy_symbols_are_working_buys_only(self):
        broker = _kabu_broker()
        with patch(_GET, return_value=_resp(_ORDERS)):
            assert broker.get_pending_buy_symbols() == {"7203", "6758"}

    def test_pending_sell_symbols_are_working_sells_only(self):
        broker = _kabu_broker()
        with patch(_GET, return_value=_resp(_ORDERS)):
            assert broker.get_pending_sell_symbols() == {"4063"}

    def test_unreadable_state_counts_as_pending(self):
        broker = _kabu_broker()
        with patch(_GET, return_value=_resp([_order("7203", "2", None)])):
            assert broker.get_pending_buy_symbols() == {"7203"}

    def test_orders_error_raises(self):
        broker = _kabu_broker()
        resp = _resp([])
        resp.raise_for_status.side_effect = RuntimeError("500")
        with patch(_GET, return_value=resp):
            with pytest.raises(RuntimeError):
                broker.get_pending_buy_symbols()

    def test_load_pending_exits_reads_kabu_sells(self):
        broker = _kabu_broker()
        with patch(_GET, return_value=_resp(_ORDERS)):
            pending = _load_pending_exits(broker)
        assert pending.has("4063", OrderSide.SELL)
        assert pending.has("4063.T", OrderSide.SELL)
        assert not pending.has("7203", OrderSide.SELL)

    def test_load_pending_exits_unreadable_gives_empty(self):
        broker = _kabu_broker()
        with patch(_GET, side_effect=RuntimeError("down")):
            pending = _load_pending_exits(broker)
        assert not pending.has("4063", OrderSide.SELL)


class TestRunDailyOrdersWithKabuPending(unittest.TestCase):
    def test_pending_buy_symbol_not_bought_again(self):
        """約定待ちの買いがある 7200 には2本目の買いを出さない。"""
        broker = _kabu_broker()
        with patch(_GET, return_value=_resp([_order("7200", "2", 3)])):
            _run(broker, _make_predictions(n_buy=2), _unique_sector)
        bought = {c.args[0] for c in broker.send_order.call_args_list}
        self.assertEqual(bought, {"7201"})

    def test_pending_buy_counts_toward_position_cap(self):
        """保有 MAX_POSITIONS − 1 件と約定待ちの買い1件で、新しく買わない。"""
        held = [{"symbol": f"S{i:03d}", "qty": 100} for i in range(MAX_POSITIONS - 1)]
        broker = _kabu_broker(positions=held)
        with patch(_GET, return_value=_resp([_order("P001", "2", 3)])):
            stats = _run(broker, _make_predictions(n_buy=3), _unique_sector)
        self.assertEqual(stats["buy_orders"], 0)
        broker.send_order.assert_not_called()

    def test_orders_unreadable_blocks_new_buy_but_exits_run(self):
        """/orders を読めない日は新規の買いを出さず、SL と売りシグナルの決済は出して失敗にする。"""
        positions = [
            {"symbol": "9000", "qty": 100, "avg_price": 1000.0, "current_price": 900.0},
            {"symbol": "9001", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0},
        ]
        broker = _kabu_broker(positions=positions)
        predictions = pd.DataFrame([_row("9001", -0.03), _row("7200", 0.03)])
        gate = MagicMock(return_value=_CORR_OK)
        with patch(_GET, side_effect=RuntimeError("orders down")):
            with self.assertRaises(RuntimeError):
                _run_with_exits(
                    broker,
                    predictions,
                    patch("src.trading.execution.runner.evaluate_correlation_gate", gate),
                )
        sent = [(c.args[0], c.args[1]) for c in broker.send_order.call_args_list]
        self.assertIn(("9000", OrderSide.SELL), sent)  # SL
        self.assertIn(("9001", OrderSide.SELL), sent)  # 売りシグナル
        self.assertNotIn(("7200", OrderSide.BUY), sent)

    def test_pending_sell_blocks_second_sell(self):
        """約定待ちの売りがある 9001 には2本目の売りを出さない。"""
        positions = [
            {"symbol": "9001", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0},
        ]
        broker = _kabu_broker(positions=positions)
        predictions = pd.DataFrame([_row("9001", -0.03)])
        gate = MagicMock(return_value=_CORR_OK)
        with patch(_GET, return_value=_resp([_order("9001", "1", 3)])):
            _run_with_exits(
                broker,
                predictions,
                patch("src.trading.execution.runner.evaluate_correlation_gate", gate),
            )
        sent = [(c.args[0], c.args[1]) for c in broker.send_order.call_args_list]
        self.assertNotIn(("9001", OrderSide.SELL), sent)
