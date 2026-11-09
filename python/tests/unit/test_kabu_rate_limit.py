"""kabu STATION® API の流量の制限の断りで待って送り直すことのテスト (#1442)。"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

import src.trading.brokers.kabu.kabu_client  # noqa: F401
import src.trading.brokers.kabu.rate_limit  # noqa: F401
from src.trading.brokers.base import OrderSide
from src.trading.brokers.kabu.kabu_client import KabuBroker
from src.trading.brokers.kabu.rate_limit import (
    RATE_LIMIT_MAX_RETRIES,
    RATE_LIMITED_STATUS,
)

_POST = "src.trading.brokers.kabu.kabu_client.httpx.post"
_GET = "src.trading.brokers.kabu.kabu_client.httpx.get"
_SLEEP = "src.trading.brokers.kabu.rate_limit._sleep"
_URL = "http://localhost:18080/kabusapi/sendorder"


def _resp(status: int, body: object = None) -> httpx.Response:
    return httpx.Response(status, json=body or {}, request=httpx.Request("POST", _URL))


def _broker() -> KabuBroker:
    broker = KabuBroker(api_password="pw")
    broker.get_token = MagicMock(return_value="tok")  # type: ignore[method-assign]
    return broker


class _LimitedServer:
    """毎秒5件を超える送りに 429 を返す偽の口。時計は待ちの関数で進む。"""

    def __init__(self, per_second: int = 5) -> None:
        self.per_second = per_second
        self.now = 0.0
        self.accepted_times: list[float] = []
        self.calls = 0

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def post(self, url: str, **kwargs: object) -> httpx.Response:
        self.calls += 1
        window = [t for t in self.accepted_times if self.now - t < 1.0]
        if len(window) >= self.per_second:
            return _resp(RATE_LIMITED_STATUS, {"Code": 4001006})
        self.accepted_times.append(self.now)
        return _resp(200, {"Result": 0, "OrderId": f"ID{len(self.accepted_times)}"})


def test_ten_sells_in_a_row_all_go_out_after_rate_limit_retry() -> None:
    server = _LimitedServer()
    broker = _broker()
    with patch(_POST, side_effect=server.post), patch(_SLEEP, side_effect=server.sleep) as sl:
        results = [broker.send_order(f"{7200 + i}", OrderSide.SELL, 100) for i in range(10)]

    assert [r["order_id"] for r in results] == [f"ID{i}" for i in range(1, 11)]
    assert len(server.accepted_times) == 10
    assert server.calls > 10  # 6件目より後は一度断られて送り直された
    assert sl.call_count >= 1
    for t in server.accepted_times:
        assert len([u for u in server.accepted_times if t <= u < t + 1.0]) <= 5


@pytest.mark.parametrize("status", [400, 401, 500, 503])
def test_non_rate_limit_error_is_not_retried(status: int) -> None:
    broker = _broker()
    with patch(_POST, return_value=_resp(status)) as post, patch(_SLEEP) as sl:
        with pytest.raises(httpx.HTTPStatusError):
            broker.send_order("7203", OrderSide.SELL, 100)
    assert post.call_count == 1
    sl.assert_not_called()


def test_rate_limit_retries_are_bounded_then_raise() -> None:
    broker = _broker()
    with patch(_POST, return_value=_resp(RATE_LIMITED_STATUS)) as post, patch(_SLEEP) as sl:
        with pytest.raises(httpx.HTTPStatusError):
            broker.send_order("7203", OrderSide.SELL, 100)
    assert post.call_count == RATE_LIMIT_MAX_RETRIES + 1
    assert sl.call_count == RATE_LIMIT_MAX_RETRIES


def test_get_balance_retries_on_rate_limit() -> None:
    broker = _broker()
    ok = _resp(200, {"StockAccountWallet": 12345.0})
    with patch(_GET, side_effect=[_resp(RATE_LIMITED_STATUS), ok]) as get, patch(_SLEEP) as sl:
        assert broker.get_balance() == 12345.0
    assert get.call_count == 2
    sl.assert_called_once()


def test_get_positions_retries_on_rate_limit() -> None:
    broker = _broker()
    ok = _resp(200, [{"Symbol": "7203", "LeavesQty": 100, "Price": 1.0, "CurrentPrice": 2.0}])
    with patch(_GET, side_effect=[_resp(RATE_LIMITED_STATUS), ok]) as get, patch(_SLEEP):
        positions = broker.get_positions()
    assert positions[0]["symbol"] == "7203"
    assert get.call_count == 2
