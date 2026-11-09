"""KabuBroker.get_orders の約定値と、live の約定差分の同期のテスト (#952)。

kabu の /orders の応答の Details のうち RecType 8 が約定の行で、Price と Qty が
その約定の値段と数量。State 5 が終了。
"""

import sys
from unittest.mock import MagicMock, patch

if "httpx" not in sys.modules:
    sys.modules["httpx"] = MagicMock()

import src.trading.brokers.kabu.kabu_client  # noqa: F401,E402
from src.trading.brokers.kabu.kabu_client import KabuBroker  # noqa: E402
from src.trading.execution.recording import _sync_live_execution_diffs  # noqa: E402


def _broker_with_orders(raw):
    broker = KabuBroker(api_password="pw")
    broker.get_token = MagicMock(return_value="tok")  # type: ignore[method-assign]
    resp = MagicMock()
    resp.json.return_value = raw
    return broker, resp


def _order(state, details, price=1000.0, cum_qty=0):
    return {
        "ID": "ORD-1",
        "Symbol": "7203",
        "Side": "2",
        "OrderQty": 300,
        "Price": price,
        "State": state,
        "CumQty": cum_qty,
        "Details": details,
    }


class TestKabuGetOrdersFillPrice:
    def test_weighted_average_of_two_execution_rows(self):
        raw = [
            _order(
                5,
                [
                    {"RecType": 1, "Price": 0, "Qty": 300},
                    {"RecType": 8, "Price": 1000.0, "Qty": 100},
                    {"RecType": 8, "Price": 1003.0, "Qty": 200},
                ],
                price=1005.0,
                cum_qty=300,
            )
        ]
        broker, resp = _broker_with_orders(raw)
        with patch("src.trading.brokers.kabu.kabu_client.httpx.get", return_value=resp):
            orders = broker.get_orders()

        assert orders[0]["fill_price"] == 1002.0
        assert orders[0]["filled_qty"] == 300
        assert orders[0]["price"] == 1005.0
        assert orders[0]["filled"] is True

    def test_no_execution_rows_gives_no_fill_price(self):
        raw = [_order(3, [{"RecType": 1, "Price": 0, "Qty": 300}])]
        broker, resp = _broker_with_orders(raw)
        with patch("src.trading.brokers.kabu.kabu_client.httpx.get", return_value=resp):
            orders = broker.get_orders()

        assert orders[0]["fill_price"] is None
        assert orders[0]["filled_qty"] == 0
        assert orders[0]["filled"] is False

    def test_missing_details_gives_no_fill_price(self):
        raw = [_order(5, None)]
        broker, resp = _broker_with_orders(raw)
        with patch("src.trading.brokers.kabu.kabu_client.httpx.get", return_value=resp):
            orders = broker.get_orders()

        assert orders[0]["fill_price"] is None
        assert orders[0]["filled"] is False

    def test_partial_fill_still_working_is_not_filled(self):
        raw = [_order(3, [{"RecType": 8, "Price": 1000.0, "Qty": 100}], cum_qty=100)]
        broker, resp = _broker_with_orders(raw)
        with patch("src.trading.brokers.kabu.kabu_client.httpx.get", return_value=resp):
            orders = broker.get_orders()

        assert orders[0]["fill_price"] == 1000.0
        assert orders[0]["filled"] is False


def _run_sync(orders):
    broker = MagicMock()
    broker.get_orders.return_value = orders
    sink = MagicMock()
    con = MagicMock()
    con.execute.return_value.fetchone.return_value = (
        "jp",
        "7203",
        "2026-11-09T08:00:00",
        1,
        1000.0,
    )
    cm = MagicMock()
    cm.__enter__.return_value = con
    with patch("src.trading.execution.recording.db_connection", return_value=cm):
        _sync_live_execution_diffs(broker, sink)
    return sink


class TestSyncLiveExecutionDiffs:
    def test_unfilled_and_cancelled_orders_are_not_written(self):
        orders = [
            {
                "order_id": "ORD-WAIT",
                "price": 1000.0,
                "status": 3,
                "fill_price": None,
                "filled_qty": 0,
                "filled": False,
            },
            {
                "order_id": "ORD-CANCEL",
                "price": 1000.0,
                "status": 5,
                "fill_price": None,
                "filled_qty": 0,
                "filled": False,
            },
        ]
        sink = _run_sync(orders)
        sink.record.assert_not_called()

    def test_filled_order_is_written_with_fill_price(self):
        orders = [
            {
                "order_id": "ORD-DONE",
                "price": 1005.0,
                "status": 5,
                "fill_price": 1002.0,
                "filled_qty": 300,
                "filled": True,
            }
        ]
        sink = _run_sync(orders)
        sink.record.assert_called_once()
        rec = sink.record.call_args[0][0]
        assert rec.actual_price == 1002.0
        assert rec.order_id == "ORD-DONE"

    def test_working_order_with_partial_fill_is_not_written(self):
        orders = [
            {
                "order_id": "ORD-PART",
                "price": 1005.0,
                "status": 3,
                "fill_price": 1000.0,
                "filled_qty": 100,
                "filled": False,
            }
        ]
        sink = _run_sync(orders)
        sink.record.assert_not_called()
