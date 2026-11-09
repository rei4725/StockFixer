"""KabuBroker.get_positions は現物の買い持ちだけを返す（#1000）"""

import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

if "httpx" not in sys.modules:
    sys.modules["httpx"] = MagicMock()

import src.trading.brokers.kabu.kabu_client  # noqa: F401,E402
from src.infrastructure.in_memory import InMemoryTradeDiffSink  # noqa: E402
from src.trading.brokers.base import OrderSide  # noqa: E402
from src.trading.execution import _check_sl_tp_exits  # noqa: E402
from src.trading.execution.selection import _get_held_symbols  # noqa: E402

_GET = "src.trading.brokers.kabu.kabu_client.httpx.get"

# 現物の買い持ち・信用の買い建て・信用の売り建て・現物で売りの行が混ざった /positions の返り
_MIXED_ROWS = [
    {
        "Symbol": "7203",
        "Side": "2",
        "LeavesQty": 100,
        "HoldQty": 0,
        "Price": 1000.0,
        "CurrentPrice": 900.0,
    },
    {
        "Symbol": "9984",
        "Side": "2",
        "MarginTradeType": 1,
        "LeavesQty": 200,
        "HoldQty": 0,
        "Price": 5000.0,
        "CurrentPrice": 4000.0,
    },
    {
        "Symbol": "6758",
        "Side": "1",
        "MarginTradeType": 2,
        "LeavesQty": 300,
        "HoldQty": 0,
        "Price": 1000.0,
        "CurrentPrice": 800.0,
    },
    {
        "Symbol": "8306",
        "Side": "1",
        "LeavesQty": 400,
        "HoldQty": 0,
        "Price": 1000.0,
        "CurrentPrice": 1500.0,
    },
]


def _kabu_broker():
    from src.trading.brokers.kabu.kabu_client import KabuBroker

    broker = KabuBroker(api_password="test_password")
    broker._token = "cached_token"
    broker._token_expires_at = datetime.now() + timedelta(hours=1)
    return broker


def _response(rows):
    resp = MagicMock()
    resp.json.return_value = rows
    resp.raise_for_status = MagicMock()
    return resp


def test_get_positions_returns_cash_long_only():
    with patch(_GET, return_value=_response(_MIXED_ROWS)):
        positions = _kabu_broker().get_positions()
    assert [p["symbol"] for p in positions] == ["7203"]
    assert positions[0]["qty"] == 100


def test_get_positions_requests_cash_product():
    with patch(_GET, return_value=_response([])) as get:
        _kabu_broker().get_positions()
    assert get.call_args.kwargs["params"]["product"] == 1


def test_held_symbols_exclude_margin_positions():
    broker = _kabu_broker()
    with patch(_GET, return_value=_response(_MIXED_ROWS)):
        held = _get_held_symbols(broker)
    assert held == {"7203"}


def test_sl_tp_exits_do_not_order_margin_positions():
    broker = _kabu_broker()
    broker.send_order = MagicMock(return_value={"order_id": "x", "status": "accepted"})
    stats = {"errors": 0, "sell_orders": 0, "short_orders": 0, "total_turnover": 0.0}
    params = {"stop_loss_pct": 0.05, "take_profit_pct": 0.10}
    with patch(_GET, return_value=_response(_MIXED_ROWS)):
        with patch("src.trading.risk_manager.get_optimal_params", return_value=params):
            with patch("src.trading.execution.sl_tp._record_order"):
                triggered = _check_sl_tp_exits(
                    broker, "jp", "kabu", MagicMock(), stats, InMemoryTradeDiffSink()
                )
    broker.send_order.assert_called_once()
    args = broker.send_order.call_args.args
    assert args[:3] == ("7203", OrderSide.SELL, 100)
    assert triggered == {"7203"}
    assert stats["sell_orders"] == 1
