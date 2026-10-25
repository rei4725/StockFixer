"""ユニットテスト: ペーパーの空売りの保有と約定を株式分割で直す（#1086）

権利落ちの日に、分割の前に建てた空売りを返済する。実 Postgres（conftest の `_isolate_db` で
ロールバック分離）を使う。株価と分割の記録はモックで、ネットにつながない。
"""

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.domain.ports import MarketDataPort
from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.trading.brokers.paper import paper_broker
from src.trading.brokers.paper.paper_broker import PaperBroker
from src.utils.db._connection import _db_connection

# 1:2 の分割の権利落ちの日
_TODAY = date(2026, 3, 10)
_SHORT = 3
_SHORT_COVER = 4


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


def _broker(open_price: float, splits: object) -> tuple[PaperBroker, InMemoryTradeDiffSink]:
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
    market_data.get_splits.return_value = splits
    sink = InMemoryTradeDiffSink()
    return PaperBroker(market_data_port=market_data, trade_diff_sink=sink), sink


def _splits(*items: tuple[str, float]) -> pd.Series:
    index = pd.DatetimeIndex([d for d, _ in items]).tz_localize("Asia/Tokyo")
    return pd.Series([r for _, r in items], index=index, name="Stock Splits")


def _add_order(
    order_id: str, side: int, qty: int, price: float = 0.0, signal_price: float | None = None
) -> None:
    with _db_connection() as con:
        con.execute(
            "INSERT INTO paper_orders (order_id, market, predicted_at, symbol, side, qty, price, "
            "signal_price, order_type, status, created_at) "
            "VALUES (%s, 'jp', %s, '7203', %s, %s, %s, %s, 10, 'pending', "
            "'2026-03-09 07:30:00')",
            [
                order_id,
                "20260309_163000" if signal_price is not None else None,
                side,
                qty,
                price,
                signal_price,
            ],
        )


def _order(order_id: str) -> tuple:
    with _db_connection() as con:
        return con.execute(
            "SELECT status, qty, fill_price, realized_pnl FROM paper_orders WHERE order_id=%s",
            [order_id],
        ).fetchone()


def _short_position() -> tuple | None:
    with _db_connection() as con:
        return con.execute(
            "SELECT qty, avg_short_price FROM paper_short_positions WHERE symbol='7203'"
        ).fetchone()


def test_short_cover_on_ex_date_covers_all_split_shares():
    broker, _ = _broker(950.0, _splits(("2026-03-10", 2.0)))
    _add_order("cover_split", _SHORT_COVER, 100)
    [fill] = broker.settle_pending_orders()
    status, qty, fill_price, realized_pnl = _order("cover_split")
    assert status == "filled"
    assert fill["qty"] == 200
    assert qty == 200
    assert fill_price == pytest.approx(950.0)
    # 売り建て 2000 を分割の比で直した 1000 から 50 円の益
    assert realized_pnl == pytest.approx((1000.0 - 950.0) * 200)
    assert _short_position() is None


def test_short_on_ex_date_adjusts_provisional_short_and_order():
    broker, _ = _broker(1050.0, _splits(("2026-03-10", 2.0)))
    # send_order の仮登録をまねる（分割前の 50 株 @ 2200）
    with _db_connection() as con:
        con.execute(
            "UPDATE paper_short_positions SET qty=50, avg_short_price=2200.0, "
            "updated_at='2026-03-09 07:30:00' WHERE symbol='7203'"
        )
    _add_order("short_split", _SHORT, 50, price=2200.0)
    broker.settle_pending_orders()
    assert _order("short_split")[1] == 100
    qty, avg = _short_position()
    assert qty == 100
    assert avg == pytest.approx(1050.0)


def test_send_short_after_split_adjusts_stored_short_before_averaging():
    broker, _ = _broker(1100.0, _splits(("2026-03-10", 2.0)))
    broker.send_order("7203", paper_broker.OrderSide.SHORT, 100, price=1100.0)
    qty, avg = _short_position()
    assert qty == 300
    assert avg == pytest.approx((1000.0 * 200 + 1100.0 * 100) / 300)


def test_unknown_splits_with_split_like_gap_keeps_short_cover_pending(caplog):
    broker, sink = _broker(950.0, None)
    _add_order("cover_unknown", _SHORT_COVER, 100)
    with caplog.at_level("WARNING"):
        assert broker.settle_pending_orders() == []
    assert _order("cover_unknown")[0] == "pending"
    assert _short_position() == (100, pytest.approx(2000.0))
    assert sink.recorded == []
    assert any("分割" in r.getMessage() for r in caplog.records)


def test_get_short_positions_adjusts_for_split():
    broker, _ = _broker(950.0, _splits(("2026-03-10", 2.0)))
    [pos] = broker.get_short_positions()
    assert pos["qty"] == 200
    assert pos["avg_short_price"] == pytest.approx(1000.0)
    assert pos["unrealized_pnl"] == pytest.approx((1000.0 - 950.0) * 200)


def test_get_short_positions_unknown_split_gap_is_not_valued():
    broker, _ = _broker(950.0, None)
    [pos] = broker.get_short_positions()
    assert pos["qty"] == 100
    assert pd.isna(pos["current_price"])
