"""ユニットテスト: ペーパーの約定を株式分割で直す（#1073）

権利落ちの日に、分割の前に出した注文を約定させる。実 Postgres（conftest の `_isolate_db` で
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


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setattr(paper_broker, "_today_jst", lambda: _TODAY)
    with _db_connection() as con:
        con.execute("DELETE FROM paper_positions")
        con.execute("DELETE FROM paper_orders")
        con.execute("UPDATE paper_balance SET balance = 1000000.0")
        con.execute(
            "INSERT INTO paper_positions (symbol, qty, avg_price, updated_at) "
            "VALUES ('7203', 100, 2000.0, '2026-03-02 01:00:00')"
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
    order_id: str,
    side: int,
    qty: int,
    price: float = 0.0,
    order_type: int = 10,
    signal_price: float | None = None,
) -> None:
    with _db_connection() as con:
        con.execute(
            "INSERT INTO paper_orders (order_id, market, predicted_at, symbol, side, qty, price, "
            "signal_price, order_type, status, created_at) "
            "VALUES (%s, 'jp', %s, '7203', %s, %s, %s, %s, %s, 'pending', "
            "'2026-03-09 07:30:00')",
            [
                order_id,
                "20260309_163000" if signal_price is not None else None,
                side,
                qty,
                price,
                signal_price,
                order_type,
            ],
        )


def _order(order_id: str) -> tuple:
    with _db_connection() as con:
        return con.execute(
            "SELECT status, qty, fill_price, realized_pnl FROM paper_orders WHERE order_id=%s",
            [order_id],
        ).fetchone()


def _position() -> tuple | None:
    with _db_connection() as con:
        return con.execute(
            "SELECT qty, avg_price FROM paper_positions WHERE symbol='7203'"
        ).fetchone()


def _balance() -> float:
    with _db_connection() as con:
        return float(con.execute("SELECT balance FROM paper_balance").fetchone()[0])


def test_market_sell_on_ex_date_sells_all_split_shares():
    broker, _ = _broker(1050.0, _splits(("2026-03-10", 2.0)))
    _add_order("sell_split", 2, 100)
    [fill] = broker.settle_pending_orders()
    status, qty, fill_price, realized_pnl = _order("sell_split")
    assert status == "filled"
    assert fill["qty"] == 200
    assert qty == 200
    assert fill_price == pytest.approx(1050.0)
    # 買値 2000 を分割の比で直した 1000 から 50 円の益
    assert realized_pnl == pytest.approx((1050.0 - 1000.0) * 200)
    assert _position() is None
    # 分割前の評価額（100 株 @ 2100 相当）と同じ水準の現金が入る
    assert _balance() == pytest.approx(1000000.0 + 1050.0 * 200)


def test_buy_on_ex_date_averages_after_split_adjustment():
    broker, _ = _broker(1100.0, _splits(("2026-03-10", 2.0)))
    _add_order("buy_split", 1, 50)
    broker.settle_pending_orders()
    # 新しい注文の株数も分割の比で直る（50 → 100）
    assert _order("buy_split")[1] == 100
    qty, avg = _position()
    assert qty == 300
    assert avg == pytest.approx((1000.0 * 200 + 1100.0 * 100) / 300)


def test_limit_price_and_signal_price_adjusted_by_split():
    broker, sink = _broker(1050.0, _splits(("2026-03-10", 2.0)))
    # 分割前の指値 2080（直すと 1040）。当日の高値は 1071 なので、直さないと失効する
    _add_order("limit_split", 2, 100, price=2080.0, order_type=20, signal_price=2080.0)
    [fill] = broker.settle_pending_orders()
    assert _order("limit_split")[0] == "filled"
    assert fill["fill_price"] == pytest.approx(1050.0)
    [record] = sink.recorded
    assert record.signal_price == pytest.approx(1040.0)
    assert record.actual_price == pytest.approx(1050.0)


def test_unknown_splits_with_split_like_gap_keeps_pending(caplog):
    broker, sink = _broker(1050.0, None)
    _add_order("sell_unknown", 2, 100, signal_price=2100.0)
    with caplog.at_level("WARNING"):
        assert broker.settle_pending_orders() == []
    assert _order("sell_unknown")[0] == "pending"
    assert _position() == (100, pytest.approx(2000.0))
    assert _balance() == pytest.approx(1000000.0)
    assert sink.recorded == []
    assert any("分割" in r.getMessage() for r in caplog.records)


def test_split_before_order_is_not_applied_to_order():
    broker, _ = _broker(1050.0, _splits(("2026-03-05", 2.0)))
    _add_order("sell_after", 2, 200)
    broker.settle_pending_orders()
    status, qty, _, realized_pnl = _order("sell_after")
    assert status == "filled"
    assert qty == 200
    assert realized_pnl == pytest.approx((1050.0 - 1000.0) * 200)
