"""live の約定差分の同期を約定の後にも呼び、約定値を1回だけ書くことのテスト (#1758)。"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

import src.orchestration.jobs.live_fill_sync  # noqa: F401
import src.trading.brokers.kabu.kabu_client  # noqa: F401
from src.orchestration.jobs import daily
from src.orchestration.jobs.live_fill_sync import sync_live_fills_after_fill
from src.trading.brokers.kabu.kabu_client import KabuBroker
from src.trading.brokers.kabu.rate_limit import RATE_LIMITED_STATUS
from src.trading.execution.recording import _sync_live_execution_diffs

_PENDING = {"order_id": "ORD-1", "status": 3, "fill_price": None, "filled_qty": 0, "filled": False}
_DONE = {"order_id": "ORD-1", "status": 5, "fill_price": 1002.0, "filled_qty": 300, "filled": True}


class _Table:
    """paper_real_diff の1行の偽。sink が書くと real_price が埋まる。"""

    def __init__(self) -> None:
        self.real_price: float | None = None

    def connection(self) -> MagicMock:
        con = MagicMock()
        con.execute.side_effect = lambda *a, **k: MagicMock(
            fetchone=MagicMock(
                return_value=("jp", "7203", "2026-11-10T08:00:00", 1, 1000.0, self.real_price)
            )
        )
        cm = MagicMock()
        cm.__enter__.return_value = con
        return cm

    def sink(self) -> MagicMock:
        sink = MagicMock()
        sink.record.side_effect = lambda rec: setattr(self, "real_price", rec.actual_price)
        return sink


def test_fill_price_is_written_once_by_the_later_sync() -> None:
    table = _Table()
    sink = table.sink()
    broker = MagicMock()
    with patch("src.trading.execution.recording.db_connection", side_effect=table.connection):
        broker.get_orders.return_value = [_PENDING]
        _sync_live_execution_diffs(broker, sink)  # 発注の直後: 未約定
        assert sink.record.call_count == 0

        broker.get_orders.return_value = [_DONE]
        _sync_live_execution_diffs(broker, sink)  # 寄りの後
        _sync_live_execution_diffs(broker, sink)  # 引けの後
    assert sink.record.call_count == 1
    assert sink.record.call_args[0][0].actual_price == 1002.0
    assert table.real_price == 1002.0


def test_sync_after_fill_uses_kabu_broker_and_trade_diff_sink() -> None:
    mod = "src.orchestration.jobs.live_fill_sync"
    with (
        patch("src.trading.brokers.kabu.kabu_client.KabuBroker") as broker_cls,
        patch(
            "src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink"
        ) as sink_cls,
        patch(f"{mod}._sync_live_execution_diffs") as sync,
    ):
        assert sync_live_fills_after_fill("settle") == 0
    sync.assert_called_once_with(broker_cls.return_value, sink_cls.return_value)


def test_sync_after_fill_failure_is_logged_not_raised() -> None:
    mod = "src.orchestration.jobs.live_fill_sync"
    with (
        patch("src.trading.brokers.kabu.kabu_client.KabuBroker"),
        patch("src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink"),
        patch(f"{mod}._sync_live_execution_diffs", side_effect=RuntimeError("x")),
        patch(f"{mod}.logger") as log,
    ):
        assert sync_live_fills_after_fill("settle") == 0
    log.error.assert_called_once()


def test_paper_settle_does_not_sync_live_fills(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTO_TRADE_MODE", "paper")
    with (
        patch.object(daily, "sync_live_fills_after_fill") as sync,
        patch("src.trading.brokers.paper.paper_broker.PaperBroker") as broker_cls,
        patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter"),
        patch("src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink"),
        patch("src.orchestration.jobs.settle_notify.notify_settle_result", return_value=0),
    ):
        broker_cls.return_value.settle_pending_orders.return_value = []
        daily.run_daily_settle_orders()
    sync.assert_not_called()


def test_kabu_get_orders_retries_on_rate_limit() -> None:
    broker = KabuBroker(api_password="pw")
    broker.get_token = MagicMock(return_value="tok")  # type: ignore[method-assign]
    req = httpx.Request("GET", "http://localhost:18080/kabusapi/orders")
    limited = httpx.Response(RATE_LIMITED_STATUS, json={}, request=req)
    ok = httpx.Response(200, json=[{"ID": "ORD-1", "State": 3, "Details": []}], request=req)
    with (
        patch("src.trading.brokers.kabu.kabu_client.httpx.get", side_effect=[limited, ok]) as get,
        patch("src.trading.brokers.kabu.rate_limit._sleep") as sl,
    ):
        orders = broker.get_orders()
    assert orders[0]["order_id"] == "ORD-1"
    assert get.call_count == 2
    sl.assert_called_once()
