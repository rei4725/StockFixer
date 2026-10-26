"""
ユニットテスト: 保有の無い SELL・SHORT_COVER を約定させない（#1190）

保有の無い決済を filled で残すと、評価額の再構成に幻の現金と建玉が入り、
返済の損益0の行が連続損失の数えを切る。
"""

from datetime import date

import pandas as pd
import pytest
from tests.unit.test_paper_broker import _make_market_data_mock

from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.trading import paper_equity
from src.trading.brokers.base import OrderSide
from src.trading.brokers.paper import paper_broker
from src.trading.brokers.paper.paper_broker import PaperBroker
from src.utils.db._connection import _db_connection

_INITIAL = 1_000_000.0


@pytest.fixture(autouse=True)
def _fixed_today(monkeypatch):
    monkeypatch.setattr(paper_broker, "_today_jst", lambda: date(2026, 3, 15))


@pytest.fixture
def broker():
    with _db_connection() as con:
        con.execute("DELETE FROM paper_orders")
        con.execute("DELETE FROM paper_positions")
        con.execute("DELETE FROM paper_short_positions")
        con.execute("UPDATE paper_balance SET balance = %s", [_INITIAL])
    return PaperBroker(
        market_data_port=_make_market_data_mock(), trade_diff_sink=InMemoryTradeDiffSink()
    )


def _order_row(order_id: str) -> tuple:
    with _db_connection() as con:
        return con.execute(
            "SELECT status, realized_pnl FROM paper_orders WHERE order_id=%s", [order_id]
        ).fetchone()


def _balance_and_holdings() -> tuple:
    with _db_connection() as con:
        balance = con.execute("SELECT balance FROM paper_balance").fetchone()[0]
        longs = con.execute("SELECT symbol, qty FROM paper_positions ORDER BY symbol").fetchall()
        shorts = con.execute(
            "SELECT symbol, qty FROM paper_short_positions ORDER BY symbol"
        ).fetchall()
    return float(balance), longs, shorts


@pytest.mark.parametrize("side", [OrderSide.SELL, OrderSide.SHORT_COVER])
def test_exit_without_position_is_not_filled(broker, side):
    """保有の無い決済は filled にせず、realized_pnl を書かず、残高と保有を動かさない"""
    before = _balance_and_holdings()
    result = broker.send_order("7203", side, 100)
    settled = broker.settle_pending_orders()

    status, pnl = _order_row(result["order_id"])
    assert status != "filled"
    assert pnl is None
    assert _balance_and_holdings() == before
    assert settled == []
    assert broker._trade_diff_sink.recorded == []


def test_equity_ignores_exits_without_position(broker):
    """保有の無い SELL・SHORT_COVER を含む履歴の評価額は、2件の無い履歴と同じ"""
    broker.send_order("7203", OrderSide.BUY, 100)
    broker.settle_pending_orders()
    expected_orders = paper_equity._load_filled_orders()
    assert len(expected_orders) == 1

    broker.send_order("6758", OrderSide.SELL, 100)
    broker.send_order("9984", OrderSide.SHORT_COVER, 100)
    broker.settle_pending_orders()
    orders = paper_equity._load_filled_orders()

    closes = {"7203": 1020.0, "6758": 1200.0, "9984": 900.0}
    now = pd.Timestamp.now().normalize()
    index = pd.date_range(now - pd.Timedelta(days=2), now + pd.Timedelta(days=1))

    def price_loader(market, symbol):
        return pd.Series(closes[symbol], index=index)

    actual = paper_equity.build_equity_series(orders, price_loader, _INITIAL, index)
    expected = paper_equity.build_equity_series(expected_orders, price_loader, _INITIAL, index)
    pd.testing.assert_series_equal(actual, expected)
