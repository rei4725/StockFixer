"""ユニットテスト: AccountGatewayBroker（口座アプリを httpx.MockTransport で偽る）"""

import json
from datetime import date

import httpx
import pytest

from src.infrastructure.in_memory import InMemoryNotificationAdapter
from src.trading.brokers.account.gateway import AccountGatewayBroker, AccountGatewayUnavailable
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


def test_timeout_stops_rest_of_day_and_notifies(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        raise httpx.ReadTimeout("slow", request=request)

    notifier = InMemoryNotificationAdapter()
    b = _broker(tmp_path, handler, notifier=notifier)
    with pytest.raises(AccountGatewayUnavailable):
        b.send_order("7203", OrderSide.BUY, 100)
    assert b.is_stopped_today()
    assert (tmp_path / "stopped-2026-10-26.json").exists()
    assert notifier.sent_alerts and notifier.sent_alerts[0]["title"] == "口座アプリに届かない"
    # 2つ目の注文は呼ばずに断る（別のジョブの Broker でも同じ印を見る）
    other = _broker(tmp_path, handler, job="rule")
    with pytest.raises(AccountGatewayUnavailable):
        other.send_order("6758", OrderSide.BUY, 100)
    assert calls == ["/v1/orders"]


def test_connection_error_stops(tmp_path):
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(AccountGatewayUnavailable):
        _broker(tmp_path, handler).get_positions()
    assert (tmp_path / "stopped-2026-10-26.json").exists()


@pytest.mark.parametrize("code", [401, 422, 500])
def test_http_error_status_stops_the_day(tmp_path, code):
    with pytest.raises(AccountGatewayUnavailable):
        _broker(tmp_path, lambda r: httpx.Response(code, json={"detail": "x"})).send_order(
            "7203", OrderSide.BUY, 100
        )
    assert (tmp_path / "stopped-2026-10-26.json").exists()


def test_unexpected_reply_stops(tmp_path):
    with pytest.raises(AccountGatewayUnavailable):
        _broker(tmp_path, lambda r: httpx.Response(200, json={"status": "maybe"})).send_order(
            "7203", OrderSide.BUY, 100
        )


def test_stop_is_per_jst_day(tmp_path):
    (tmp_path / "stopped-2026-10-23.json").write_text("{}", encoding="utf-8")
    b = _broker(tmp_path, lambda r: httpx.Response(200, json={"status": "accepted", "order_id": 1}))
    assert not b.is_stopped_today()
    assert b.send_order("7203", OrderSide.BUY, 100)["status"] == "pending"


def test_notifier_failure_does_not_hide_the_stop(tmp_path):
    class Broken(InMemoryNotificationAdapter):
        def send_alert(self, title, message, level=None):
            raise RuntimeError("discord down")

    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(AccountGatewayUnavailable):
        _broker(tmp_path, handler, notifier=Broken()).send_order("7203", OrderSide.BUY, 100)
    assert (tmp_path / "stopped-2026-10-26.json").exists()
