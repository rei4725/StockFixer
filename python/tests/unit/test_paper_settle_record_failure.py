"""ユニットテスト: 確定の後の paper_real_diff の記録が落ちても、注文は約定に数える（#1573）

残高・保有・注文は確定済みで戻らないので、記録の失敗は ERROR に出すだけにする。
errors（last_settle_errors）は「例外で約定できず pending に残した件数」のままにする。
実 Postgres（conftest の `_isolate_db` でロールバック分離）を使う。
"""

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.domain.ports import MarketDataPort, TradeDiffSink
from src.trading.brokers.paper import paper_broker
from src.trading.brokers.paper.paper_broker import PaperBroker
from src.utils.db import _connection

_TODAY = date(2026, 3, 10)
_OPEN = 1000.0
_INITIAL = 1000000.0
_FAILED_ORDER = "fail1"
_NEXT_ORDER = "next1"


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setattr(paper_broker, "_today_jst", lambda: _TODAY)
    with _connection._db_connection() as con:
        con.execute("DELETE FROM paper_positions")
        con.execute("DELETE FROM paper_orders")
        con.execute("UPDATE paper_balance SET balance = %s", [_INITIAL])


def _broker(sink: MagicMock) -> PaperBroker:
    market_data = MagicMock(spec=MarketDataPort)
    market_data.get_ohlcv.return_value = pd.DataFrame(
        {"Open": [_OPEN], "High": [_OPEN * 1.02], "Low": [_OPEN * 0.98], "Close": [_OPEN]},
        index=pd.to_datetime(["2026-03-10"]),
    )
    market_data.get_splits.return_value = pd.Series(
        [], index=pd.DatetimeIndex([]).tz_localize("Asia/Tokyo"), dtype=float
    )
    return PaperBroker(market_data_port=market_data, trade_diff_sink=sink)


def _add_position(symbol: str, qty: int = 100, avg: float = 900.0) -> None:
    with _connection._db_connection() as con:
        con.execute(
            "INSERT INTO paper_positions (symbol, qty, avg_price, updated_at) "
            "VALUES (%s, %s, %s, '2026-03-02 01:00:00')",
            [symbol, qty, avg],
        )


def _add_order(order_id: str, symbol: str, side: int, created_at: str) -> None:
    with _connection._db_connection() as con:
        con.execute(
            "INSERT INTO paper_orders (order_id, market, predicted_at, symbol, side, qty, price, "
            "signal_price, order_type, status, created_at) "
            "VALUES (%s, 'jp', '20260309_163000', %s, %s, 100, 0, 990.0, 10, 'pending', %s)",
            [order_id, symbol, side, created_at],
        )


def _balance() -> float:
    with _connection._db_connection() as con:
        return float(con.execute("SELECT balance FROM paper_balance").fetchone()[0])


def _position_qty(symbol: str) -> int | None:
    with _connection._db_connection() as con:
        row = con.execute("SELECT qty FROM paper_positions WHERE symbol=%s", [symbol]).fetchone()
    return int(row[0]) if row else None


def _order_status(order_id: str) -> str:
    with _connection._db_connection() as con:
        return str(
            con.execute("SELECT status FROM paper_orders WHERE order_id=%s", [order_id]).fetchone()[
                0
            ]
        )


def _failing_sink() -> MagicMock:
    """1件目の記録だけ投げる Sink"""
    sink = MagicMock(spec=TradeDiffSink)
    sink.record.side_effect = [RuntimeError("記録の失敗"), None]
    return sink


def _settle_with_failed_record(caplog) -> tuple[PaperBroker, list[dict], list[str]]:
    sink = _failing_sink()
    broker = _broker(sink)
    with caplog.at_level("ERROR"):
        settled = broker.settle_pending_orders()
    assert sink.record.call_count == 2
    return broker, settled, [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]


def _assert_counted_as_filled(broker: PaperBroker, settled: list[dict]) -> None:
    assert [fill["order_id"] for fill in settled] == [_FAILED_ORDER, _NEXT_ORDER]
    assert broker.last_settle_errors == 0
    assert _order_status(_FAILED_ORDER) == "filled"
    assert _order_status(_NEXT_ORDER) == "filled"


def test_buy_counts_as_filled_when_record_fails(caplog):
    _add_order(_FAILED_ORDER, "7203", 1, "2026-03-09 07:30:00")
    _add_order(_NEXT_ORDER, "6758", 1, "2026-03-09 07:31:00")

    broker, settled, errors = _settle_with_failed_record(caplog)

    _assert_counted_as_filled(broker, settled)
    assert _balance() == pytest.approx(_INITIAL - 2 * _OPEN * 100)
    assert _position_qty("7203") == 100
    assert _position_qty("6758") == 100
    assert any(_FAILED_ORDER in m for m in errors)
    assert not any("pending に据え置いた" in m for m in errors)


def test_sell_counts_as_filled_when_record_fails(caplog):
    _add_position("7203")
    _add_order(_FAILED_ORDER, "7203", 2, "2026-03-09 07:30:00")
    _add_order(_NEXT_ORDER, "6758", 1, "2026-03-09 07:31:00")

    broker, settled, errors = _settle_with_failed_record(caplog)

    _assert_counted_as_filled(broker, settled)
    assert _balance() == pytest.approx(_INITIAL)  # 売りの入金と買いの出金が同じ額
    assert _position_qty("7203") is None
    assert _position_qty("6758") == 100
    assert any(_FAILED_ORDER in m for m in errors)
