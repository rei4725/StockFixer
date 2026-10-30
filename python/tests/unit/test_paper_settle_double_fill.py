"""ユニットテスト: 2つの約定の回が同じ注文を二重に約定させない（#1419）

9:05 の約定の補完と約定のやり直しが重なると、どちらも同じ注文を pending として読む。
1回目が株価を引いている間（読みの後、書き込みの前）に2回目が読んで約定させる順を再現し、
残高・保有・注文が1回分だけ動くことを確かめる。実 Postgres（conftest の `_isolate_db`）を使う。
"""

from contextlib import contextmanager
from datetime import date
from typing import Any, Iterator
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.domain.ports import MarketDataPort
from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.trading.brokers.paper import paper_broker
from src.trading.brokers.paper.paper_broker import PaperBroker
from src.utils.db import _connection

_TODAY = date(2026, 3, 10)
_OPEN = 1000.0


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    """接続を、自動確定の本番と同じく SAVEPOINT を切らない形にする"""

    @contextmanager
    def _plain_connection(lock_timeout: float | None = None) -> Iterator[Any]:
        yield _connection._test_connection

    with _connection._db_connection() as con:
        con.execute("DELETE FROM paper_positions")
        con.execute("DELETE FROM paper_orders")
        con.execute("UPDATE paper_balance SET balance = 1000000.0")
    monkeypatch.setattr(paper_broker, "_db_connection", _plain_connection)
    monkeypatch.setattr(paper_broker, "_today_jst", lambda: _TODAY)


def _market_data() -> MagicMock:
    market_data = MagicMock(spec=MarketDataPort)
    market_data.get_ohlcv.return_value = pd.DataFrame(
        {"Open": [_OPEN], "High": [_OPEN * 1.02], "Low": [_OPEN * 0.98], "Close": [_OPEN]},
        index=pd.to_datetime(["2026-03-10"]),
    )
    market_data.get_splits.return_value = pd.Series(
        [], index=pd.DatetimeIndex([]).tz_localize("Asia/Tokyo"), dtype=float
    )
    return market_data


def _overlapping_brokers() -> tuple[PaperBroker, PaperBroker]:
    """1回目の株価の取得の中で、2回目の回を最後まで走らせる"""
    second = PaperBroker(market_data_port=_market_data(), trade_diff_sink=InMemoryTradeDiffSink())
    first_md = _market_data()
    bars = first_md.get_ohlcv.return_value

    def _get_ohlcv(*args: Any, **kwargs: Any) -> pd.DataFrame:
        if first_md.get_ohlcv.call_count == 1:
            second.settle_pending_orders()
        return bars

    first_md.get_ohlcv.side_effect = _get_ohlcv
    first = PaperBroker(market_data_port=first_md, trade_diff_sink=InMemoryTradeDiffSink())
    return first, second


def _add_position(qty: int = 100, avg: float = 900.0) -> None:
    with _connection._db_connection() as con:
        con.execute(
            "INSERT INTO paper_positions (symbol, qty, avg_price, updated_at) "
            "VALUES ('7203', %s, %s, '2026-03-02 01:00:00')",
            [qty, avg],
        )


def _add_order(order_id: str, side: int, order_type: int = 10, price: float = 0.0) -> None:
    with _connection._db_connection() as con:
        con.execute(
            "INSERT INTO paper_orders (order_id, market, predicted_at, symbol, side, qty, price, "
            "order_type, status, created_at) "
            "VALUES (%s, 'jp', NULL, '7203', %s, 100, %s, %s, 'pending', '2026-03-09 07:30:00')",
            [order_id, side, price, order_type],
        )


def _state(order_id: str) -> tuple:
    with _connection._db_connection() as con:
        balance = float(con.execute("SELECT balance FROM paper_balance").fetchone()[0])
        position = con.execute(
            "SELECT qty, avg_price FROM paper_positions WHERE symbol='7203'"
        ).fetchone()
        order = con.execute(
            "SELECT status, qty, fill_price, realized_pnl FROM paper_orders WHERE order_id=%s",
            [order_id],
        ).fetchone()
    return balance, position, order


def test_overlapping_buy_fills_once():
    _add_order("buy1", 1)
    first, _ = _overlapping_brokers()
    settled = first.settle_pending_orders()
    balance, position, order = _state("buy1")
    assert balance == pytest.approx(1000000.0 - _OPEN * 100)
    assert position == (100, pytest.approx(_OPEN))
    assert order[0] == "filled"
    assert order[1] == 100
    assert settled == []


def test_overlapping_sell_fills_once():
    _add_position(qty=200)
    _add_order("sell1", 2)
    first, _ = _overlapping_brokers()
    settled = first.settle_pending_orders()
    balance, position, order = _state("sell1")
    assert balance == pytest.approx(1000000.0 + _OPEN * 100)
    assert position == (100, pytest.approx(900.0))
    assert order[0] == "filled"
    assert order[1] == 100
    assert order[3] == pytest.approx((_OPEN - 900.0) * 100)
    assert settled == []


def test_overlapping_expired_limit_is_not_overwritten():
    """失効の後に別の回が読んだ注文を、もう一度失効させない（SHORT の仮登録の二重の戻しを防ぐ）"""
    _add_order("buy_limit", 1, order_type=20, price=_OPEN * 0.5)
    first, _ = _overlapping_brokers()
    assert first.settle_pending_orders() == []
    balance, position, order = _state("buy_limit")
    assert balance == pytest.approx(1000000.0)
    assert position is None
    assert order[0] == "expired"
