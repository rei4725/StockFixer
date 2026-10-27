"""ユニットテスト: ペーパーの空売りの指値が失効したら、send_order の仮登録を戻す（#1219）

実 Postgres（conftest の `_isolate_db` でロールバック分離）を使う。株価と分割の記録はモックで、
ネットにつながない。
"""

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.domain.ports import MarketDataPort
from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.trading.brokers.base import OrderSide, OrderType
from src.trading.brokers.paper import paper_broker
from src.trading.brokers.paper.paper_broker import PaperBroker
from src.utils.db._connection import _db_connection

_TODAY = date(2026, 3, 10)


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setattr(paper_broker, "_today_jst", lambda: _TODAY)
    with _db_connection() as con:
        con.execute("DELETE FROM paper_positions")
        con.execute("DELETE FROM paper_short_positions")
        con.execute("DELETE FROM paper_orders")
        con.execute("UPDATE paper_balance SET balance = 1000000.0")


def _broker(open_price: float, high_price: float, splits: object) -> PaperBroker:
    market_data = MagicMock(spec=MarketDataPort)
    market_data.get_ohlcv.return_value = pd.DataFrame(
        {
            "Open": [open_price],
            "High": [high_price],
            "Low": [open_price * 0.98],
            "Close": [open_price],
        },
        index=pd.to_datetime(["2026-03-10"]),
    )
    market_data.get_splits.return_value = splits
    return PaperBroker(market_data_port=market_data, trade_diff_sink=InMemoryTradeDiffSink())


def _no_splits() -> pd.Series:
    return pd.Series([], index=pd.DatetimeIndex([]).tz_localize("Asia/Tokyo"), dtype=float)


def _short_position() -> tuple | None:
    with _db_connection() as con:
        return con.execute(
            "SELECT qty, avg_short_price FROM paper_short_positions WHERE symbol='7203'"
        ).fetchone()


def _status(order_id: str) -> str:
    with _db_connection() as con:
        return con.execute(
            "SELECT status FROM paper_orders WHERE order_id=%s", [order_id]
        ).fetchone()[0]


def test_expired_short_limit_without_prior_short_leaves_no_row():
    broker = _broker(950.0, 969.0, _no_splits())
    sent = broker.send_order("7203", OrderSide.SHORT, 100, price=1000.0, order_type=OrderType.LIMIT)
    assert _short_position() == (100, pytest.approx(1000.0))
    assert broker.settle_pending_orders() == []
    assert _status(sent["order_id"]) == "expired"
    assert _short_position() is None


def test_expired_short_limit_restores_prior_short():
    with _db_connection() as con:
        con.execute(
            "INSERT INTO paper_short_positions "
            "(symbol, qty, avg_short_price, opened_at, updated_at) "
            "VALUES ('7203', 100, 900.0, '2026-03-02 01:00:00', '2026-03-02 01:00:00')"
        )
    broker = _broker(950.0, 969.0, _no_splits())
    sent = broker.send_order("7203", OrderSide.SHORT, 50, price=1000.0, order_type=OrderType.LIMIT)
    assert _short_position()[0] == 150
    assert broker.settle_pending_orders() == []
    assert _status(sent["order_id"]) == "expired"
    assert _short_position() == (100, pytest.approx(900.0))


def test_expired_short_limit_on_ex_date_undoes_split_adjusted_registration():
    # 100 株 @ 2000 に、分割の前の日に 50 株 @ 2200 の指値の空売りを仮登録した行
    with _db_connection() as con:
        con.execute(
            "INSERT INTO paper_short_positions "
            "(symbol, qty, avg_short_price, opened_at, updated_at) "
            "VALUES ('7203', 150, %s, '2026-03-02 01:00:00', '2026-03-09 07:30:00')",
            [(2000.0 * 100 + 2200.0 * 50) / 150],
        )
        con.execute(
            "INSERT INTO paper_orders (order_id, market, symbol, side, qty, price, "
            "order_type, status, created_at) "
            "VALUES ('short_split', 'jp', '7203', %s, 50, 2200.0, %s, 'pending', "
            "'2026-03-09 07:30:00')",
            [int(OrderSide.SHORT), int(OrderType.LIMIT)],
        )
    index = pd.DatetimeIndex(["2026-03-10"]).tz_localize("Asia/Tokyo")
    # 1:2 の権利落ち。直った指値 1100 に高値 1071 が届かず失効する
    broker = _broker(1050.0, 1071.0, pd.Series([2.0], index=index))
    assert broker.settle_pending_orders() == []
    assert _status("short_split") == "expired"
    assert _short_position() == (200, pytest.approx(1000.0))
