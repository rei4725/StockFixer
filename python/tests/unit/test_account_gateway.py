"""ユニットテスト: AccountGatewayBroker（口座アプリを httpx.MockTransport で偽る）"""

import json
from datetime import date

import httpx
import pytest

from src.trading.brokers.account.gateway import AccountGatewayBroker
from src.trading.brokers.base import BrokerError, OrderSide, OrderType

DAY = date(2026, 10, 26)


def _broker(tmp_path, handler, job="auto", notifier=None):
    return AccountGatewayBroker(
        "http://account:8000",
        "tok",
        job=job,
        state_dir=str(tmp_path),
        notifier=notifier,
        transport=httpx.MockTransport(handler),
        today=lambda: DAY,
    )


def test_send_order_accepted_maps_to_pending(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "accepted", "order_id": 7})

    result = _broker(tmp_path, handler).send_order("7203.T", OrderSide.BUY, 100)
    assert result == {"order_id": "7", "status": "pending", "message": "accepted"}
    assert seen["auth"] == "Bearer tok"
    assert seen["body"] == {
        "client_order_id": "20261026-auto-7203-buy",
        "symbol": "7203",
        "market": "jp",
        "side": "buy",
        "qty": 100,
        "order_type": "market",
        "reason": "stockfixer:auto",
    }


def test_limit_order_sends_limit_price(tmp_path):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "accepted", "order_id": 1})

    _broker(tmp_path, handler).send_order(
        "7203", OrderSide.SELL, 100, price=2500.0, order_type=OrderType.LIMIT
    )
    assert seen["body"]["order_type"] == "limit"
    assert seen["body"]["limit_price"] == 2500.0
    assert seen["body"]["side"] == "sell"


def test_rejected_keeps_reason_code(tmp_path):
    def handler(request):
        return httpx.Response(
            200, json={"status": "rejected", "reason_code": "max_positions", "detail": "10銘柄"}
        )

    result = _broker(tmp_path, handler).send_order("7203", OrderSide.BUY, 100)
    assert result == {
        "order_id": None,
        "status": "rejected",
        "reason_code": "max_positions",
        "message": "max_positions: 10銘柄",
    }


def test_client_order_id_is_deterministic(tmp_path):
    b = _broker(tmp_path, lambda r: httpx.Response(200, json={}), job="rule")
    assert b.client_order_id("6758.T", OrderSide.SELL) == "20261026-rule-6758-sell"
    assert b.client_order_id("6758", OrderSide.SELL) == b.client_order_id("6758.T", OrderSide.SELL)


def test_short_side_is_refused_without_calling(tmp_path):
    def handler(request):
        raise AssertionError("呼んではならない")

    with pytest.raises(BrokerError):
        _broker(tmp_path, handler).send_order("7203", OrderSide.SHORT, 100)


def test_positions_balance_orders(tmp_path):
    def handler(request):
        if request.url.path == "/v1/positions":
            return httpx.Response(
                200,
                json=[
                    {
                        "symbol": "7203",
                        "market": "jp",
                        "qty": 100,
                        "avg_price": 2500.0,
                        "last_price": 2550.0,
                    },
                    {
                        "symbol": "AAPL",
                        "market": "us",
                        "qty": 1,
                        "avg_price": 200.0,
                        "last_price": 210.0,
                    },
                ],
            )
        if request.url.path == "/v1/balance":
            return httpx.Response(200, json={"cash": 640000.0, "equity": 895000.0})
        if request.url.path == "/v1/orders":
            assert request.url.params["date"] == "2026-10-26"
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 3,
                        "symbol": "7203",
                        "side": "buy",
                        "qty": 100,
                        "limit_price": None,
                        "status": "pending",
                    }
                ],
            )
        return httpx.Response(404)

    b = _broker(tmp_path, handler)
    assert b.get_positions() == [
        {"symbol": "7203", "market": "jp", "qty": 100, "avg_price": 2500.0, "current_price": 2550.0}
    ]
    assert b.get_balance() == 640000.0
    assert b.get_equity() == 895000.0
    assert b.get_orders() == [
        {
            "order_id": "3",
            "symbol": "7203",
            "side": "buy",
            "qty": 100,
            "price": None,
            "status": "pending",
        }
    ]
    assert b.get_orders_on(DAY)[0]["id"] == 3


def test_get_token_calls_health(tmp_path):
    def handler(request):
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok"})

    assert _broker(tmp_path, handler).get_token() == "account"


def test_cancel_is_not_supported(tmp_path):
    with pytest.raises(BrokerError):
        _broker(tmp_path, lambda r: httpx.Response(200, json={})).cancel_order("1")
