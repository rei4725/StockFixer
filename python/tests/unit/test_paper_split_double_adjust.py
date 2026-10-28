"""ユニットテスト: 権利落ちの日の寄りの前に直した保有を、同じ分割で二度直さない（#1326）

DB の updated_at と created_at は tz 無しの UTC。権利落ちの日 D の 8:50 JST は UTC で D-1 23:50 になる。
実 Postgres（conftest の `_isolate_db` でロールバック分離）を使う。株価と分割の記録はモック。
"""

from datetime import date, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.domain.ports import MarketDataPort
from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.trading.brokers.base import OrderSide
from src.trading.brokers.paper import paper_broker
from src.trading.brokers.paper import split_adjust as sa
from src.trading.brokers.paper.paper_broker import PaperBroker
from src.utils.db._connection import _db_connection

# 1:2 の分割の権利落ちの日 D
_TODAY = date(2026, 3, 10)
# D の 8:50 JST（tz 無しの UTC）
_D_0850_JST = datetime(2026, 3, 9, 23, 50)
# D-1 の 16:30 JST（tz 無しの UTC）
_D_PREV_1630_JST = datetime(2026, 3, 9, 7, 30)


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setattr(paper_broker, "_today_jst", lambda: _TODAY)
    monkeypatch.setattr(sa, "_now_utc", lambda: _D_0850_JST)
    with _db_connection() as con:
        con.execute("DELETE FROM paper_positions")
        con.execute("DELETE FROM paper_short_positions")
        con.execute("DELETE FROM paper_orders")


def _market_data() -> MagicMock:
    market_data = MagicMock(spec=MarketDataPort)
    market_data.get_ohlcv.return_value = pd.DataFrame(
        {"Open": [990.0], "High": [990.0], "Low": [990.0], "Close": [990.0]},
        index=pd.to_datetime(["2026-03-09"]),
    )
    index = pd.DatetimeIndex(["2026-03-10"]).tz_localize("Asia/Tokyo")
    market_data.get_splits.return_value = pd.Series([2.0], index=index, name="Stock Splits")
    return market_data


def _broker() -> PaperBroker:
    return PaperBroker(market_data_port=_market_data(), trade_diff_sink=InMemoryTradeDiffSink())


def test_long_position_adjusted_before_open_is_not_adjusted_again():
    with _db_connection() as con:
        con.execute(
            "INSERT INTO paper_positions (symbol, qty, avg_price, updated_at) "
            "VALUES ('7203', 100, 2000.0, '2026-03-02 01:00:00')"
        )
    broker = _broker()
    [first] = broker.get_positions()
    assert (first["qty"], first["avg_price"]) == (200, pytest.approx(1000.0))
    [again] = broker.get_positions()
    assert again["qty"] == 200
    assert again["avg_price"] == pytest.approx(1000.0)


def test_short_position_adjusted_before_open_is_not_adjusted_again():
    with _db_connection() as con:
        con.execute(
            "INSERT INTO paper_short_positions "
            "(symbol, qty, avg_short_price, opened_at, updated_at) "
            "VALUES ('7203', 100, 2000.0, '2026-03-02 01:00:00', '2026-03-02 01:00:00')"
        )
    broker = _broker()
    [first] = broker.get_short_positions()
    assert (first["qty"], first["avg_short_price"]) == (200, pytest.approx(1000.0))
    [again] = broker.get_short_positions()
    assert again["qty"] == 200
    assert again["avg_short_price"] == pytest.approx(1000.0)


@pytest.mark.parametrize("created_at", [_D_PREV_1630_JST, _D_0850_JST])
def test_order_created_before_open_is_adjusted_by_split_on_ex_date(created_at):
    order = (int(OrderSide.BUY), 100, 2000.0, 1990.0, created_at)
    adjusted = sa.adjust_order_for_splits(_market_data(), "7203", order, 1000.0)
    assert adjusted is not None
    qty, limit_price, signal_price = adjusted
    assert qty == 200
    assert limit_price == pytest.approx(1000.0)
    assert signal_price == pytest.approx(995.0)
