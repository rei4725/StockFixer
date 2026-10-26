"""ユニットテスト: 分割をまたぐペーパーの空売りの重ね建ての約定（#1130）

空売りの保有に重ね建ての注文を出し、約定の前に権利落ちした分割で保有と注文の株数が直っても、
売り建ての単価が分割後の前の保有と約定の加重平均になることを確かめる。
実 Postgres（conftest の `_isolate_db` でロールバック分離）を使う。株価と分割の記録はモック。
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

# 1:2 の分割の権利落ちの日
_TODAY = date(2026, 3, 10)


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setattr(paper_broker, "_today_jst", lambda: _TODAY)
    with _db_connection() as con:
        con.execute("DELETE FROM paper_positions")
        con.execute("DELETE FROM paper_short_positions")
        con.execute("DELETE FROM paper_orders")
        con.execute("UPDATE paper_balance SET balance = 1000000.0")
        con.execute(
            "INSERT INTO paper_short_positions "
            "(symbol, qty, avg_short_price, opened_at, updated_at) "
            "VALUES ('7203', 100, 2000.0, '2026-03-02 01:00:00', '2026-03-02 01:00:00')"
        )


def _broker(open_price: float) -> tuple[PaperBroker, MagicMock]:
    market_data = MagicMock(spec=MarketDataPort)
    market_data.get_ohlcv.return_value = pd.DataFrame(
        {
            "Open": [open_price],
            "High": [open_price * 1.02],
            "Low": [open_price * 0.98],
            "Close": [open_price],
        },
        index=pd.to_datetime(["2026-03-10"]),
    )
    # 発注の時点では分割の記録はまだ無い
    market_data.get_splits.return_value = pd.Series([], dtype=float, name="Stock Splits")
    broker = PaperBroker(market_data_port=market_data, trade_diff_sink=InMemoryTradeDiffSink())
    return broker, market_data


def _record_split(market_data: MagicMock) -> None:
    index = pd.DatetimeIndex(["2026-03-10"]).tz_localize("Asia/Tokyo")
    market_data.get_splits.return_value = pd.Series([2.0], index=index, name="Stock Splits")


def _send_short_before_ex_date(broker: PaperBroker, price: float, order_type: OrderType) -> str:
    result = broker.send_order("7203", OrderSide.SHORT, 50, price=price, order_type=order_type)
    order_id = str(result["order_id"])
    # 発注と仮登録を権利落ちの前の日にずらす
    with _db_connection() as con:
        con.execute(
            "UPDATE paper_orders SET created_at='2026-03-09 07:30:00' WHERE order_id=%s",
            [order_id],
        )
        con.execute(
            "UPDATE paper_short_positions SET updated_at='2026-03-09 07:30:00' "
            "WHERE symbol='7203'"
        )
    return order_id


def _short_position() -> tuple:
    with _db_connection() as con:
        return con.execute(
            "SELECT qty, avg_short_price FROM paper_short_positions WHERE symbol='7203'"
        ).fetchone()


def test_market_short_add_across_split_averages_prior_holding_and_fill():
    broker, market_data = _broker(1050.0)
    _send_short_before_ex_date(broker, 0.0, OrderType.MARKET)
    _record_split(market_data)
    [fill] = broker.settle_pending_orders()
    assert fill["qty"] == 100
    qty, avg = _short_position()
    assert qty == 300
    # 分割後の前の保有 200株 @ 1000 と、約定 100株 @ 1050 の加重平均
    assert avg == pytest.approx((1000.0 * 200 + 1050.0 * 100) / 300)


def test_limit_short_add_across_split_averages_prior_holding_and_fill():
    broker, market_data = _broker(1060.0)
    _send_short_before_ex_date(broker, 2100.0, OrderType.LIMIT)
    _record_split(market_data)
    [fill] = broker.settle_pending_orders()
    assert fill["qty"] == 100
    qty, avg = _short_position()
    assert qty == 300
    # 指値 2100 は分割で 1050 に直り、始値 1060 で約定する
    assert avg == pytest.approx((1000.0 * 200 + 1060.0 * 100) / 300)
