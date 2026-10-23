"""約束のテスト: 本体が送る形と受ける形が contracts/account-api.json に合うか

偽の口座アプリは、届いた願いを OrderRequest の形で確かめ、返す答えも OrderReply の形で確かめてから返す。
約束のファイルが変わって本体の送る形がずれると、ここが落ちる。
"""

import json
from datetime import date
from pathlib import Path

import httpx
import pytest
from jsonschema import Draft202012Validator

from src.trading.brokers.account.gateway import AccountGatewayBroker
from src.trading.brokers.base import OrderSide, OrderType

CONTRACT = json.loads(
    (Path(__file__).resolve().parents[3] / "contracts" / "account-api.json").read_text(
        encoding="utf-8"
    )
)
REASON_CODES = [
    "daily_loss",
    "position_size",
    "max_positions",
    "max_orders",
    "halted",
    "invalid",
    "internal_error",
]


def _validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(
        {"$ref": f"#/components/schemas/{name}", "components": CONTRACT["components"]}
    )


REQUEST = _validator("OrderRequest")
REPLY = _validator("OrderReply")


def _fake_account(reply: dict):
    REPLY.validate(reply)  # 偽の答えも約束の形であること

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/orders"
        REQUEST.validate(json.loads(request.content))
        return httpx.Response(200, json=reply)

    return handler


def _broker(tmp_path, handler):
    return AccountGatewayBroker(
        "http://account:8000",
        "tok",
        job="auto",
        state_dir=str(tmp_path),
        transport=httpx.MockTransport(handler),
        today=lambda: date(2026, 10, 26),
    )


def test_contract_reason_codes_match_ours():
    props = CONTRACT["components"]["schemas"]["OrderReply"]["properties"]
    assert props["reason_code"]["anyOf"][0]["enum"] == REASON_CODES


@pytest.mark.parametrize(
    "side,order_type,price",
    [(OrderSide.BUY, OrderType.MARKET, 0.0), (OrderSide.SELL, OrderType.LIMIT, 2500.0)],
)
def test_request_matches_contract(tmp_path, side, order_type, price):
    b = _broker(tmp_path, _fake_account({"status": "accepted", "order_id": 1}))
    result = b.send_order("7203.T", side, 100, price=price, order_type=order_type)
    assert result["status"] == "pending"


@pytest.mark.parametrize("code", REASON_CODES)
def test_every_rejection_code(tmp_path, code):
    b = _broker(tmp_path, _fake_account({"status": "rejected", "reason_code": code, "detail": "d"}))
    result = b.send_order("7203", OrderSide.BUY, 100)
    assert result["status"] == "rejected" and result["reason_code"] == code
    assert not b.is_stopped_today()  # 断りは止めない（届かないときだけ止める）


def test_client_order_id_fits_contract_length(tmp_path):
    b = _broker(tmp_path, _fake_account({"status": "accepted", "order_id": 1}))
    cid = b.client_order_id("ABCDEFGHIJKLMNOP", OrderSide.SELL)
    props = CONTRACT["components"]["schemas"]["OrderRequest"]["properties"]
    assert len(cid) <= props["client_order_id"]["maxLength"]


def test_paths_we_call_exist_in_contract():
    for path in ["/v1/orders", "/v1/positions", "/v1/balance", "/health"]:
        assert path in CONTRACT["paths"], path
