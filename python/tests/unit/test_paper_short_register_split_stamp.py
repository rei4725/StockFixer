"""ユニットテスト: 直した空売りの保有へ SHORT を重ねても、同じ分割を二度当てない（#1339）

DB の updated_at と created_at は tz 無しの UTC で、日付のまま読まれる。権利落ちの日 D を
UTC の今日の翌日に置くと、今の CURRENT_TIMESTAMP は D の 9:00 JST より前の時刻（D-1 と読まれる）になる。
実 Postgres（conftest の `_isolate_db` でロールバック分離）を使う。株価と分割の記録はモック。
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.domain.ports import MarketDataPort
from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.trading.brokers.base import OrderSide, OrderType
from src.trading.brokers.paper.paper_broker import PaperBroker
from src.utils.db._connection import _db_connection


@pytest.fixture(autouse=True)
def _setup():
    with _db_connection() as con:
        con.execute("DELETE FROM paper_positions")
        con.execute("DELETE FROM paper_short_positions")
        con.execute("DELETE FROM paper_orders")
        con.execute(
            "INSERT INTO paper_short_positions "
            "(symbol, qty, avg_short_price, opened_at, updated_at) "
            "VALUES ('7203', 100, 2000.0, '2026-03-02 01:00:00', '2026-03-02 01:00:00')"
        )


def _broker() -> PaperBroker:
    ex_date = (datetime.now(timezone.utc) + timedelta(days=1)).date()
    market_data = MagicMock(spec=MarketDataPort)
    market_data.get_ohlcv.return_value = pd.DataFrame(
        {"Open": [990.0], "High": [990.0], "Low": [990.0], "Close": [990.0]},
        index=pd.to_datetime([ex_date - timedelta(days=1)]),
    )
    index = pd.DatetimeIndex([str(ex_date)]).tz_localize("Asia/Tokyo")
    market_data.get_splits.return_value = pd.Series([2.0], index=index, name="Stock Splits")
    return PaperBroker(market_data_port=market_data, trade_diff_sink=InMemoryTradeDiffSink())


def _short_on_adjusted_holding(broker: PaperBroker) -> str:
    [first] = broker.get_short_positions()
    assert (first["qty"], first["avg_short_price"]) == (200, pytest.approx(1000.0))
    order = broker.send_order("7203", OrderSide.SHORT, 10, 1990.0, OrderType.LIMIT)
    return str(order["order_id"])


def test_short_added_after_adjustment_does_not_adjust_old_part_again():
    broker = _broker()
    _short_on_adjusted_holding(broker)
    [again] = broker.get_short_positions()
    # 前からの分は 200 株のまま。新しい 10 株（分割の前の値段）は分割を当てて 20 株に数える
    assert again["qty"] == 220
    assert again["pending_qty"] == 20
    assert again["avg_short_price"] == pytest.approx(1000.0)


def test_cancel_after_adding_short_does_not_adjust_old_part_again():
    broker = _broker()
    order_id = _short_on_adjusted_holding(broker)
    broker.cancel_order(order_id)
    [again] = broker.get_short_positions()
    assert again["qty"] == 200
    assert again["pending_qty"] == 0
    assert again["avg_short_price"] == pytest.approx(1000.0)
