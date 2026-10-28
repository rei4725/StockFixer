"""ユニットテスト: 足の日の引けの後に出た注文を、その足で約定させない（#1312）

9:05 の約定が遅れて 16:30 のルールの発注の後に走ると、その注文を同じ日の（注文より前の）
始値で約定させていた。注文の created_at を JST に直し、足の日の東証の引け（15:30）以後に
出た注文は pending に据え置く。実 Postgres（conftest の `_isolate_db` でロールバック分離）を使う。
"""

import logging
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.domain.ports import MarketDataPort
from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.trading.brokers.base import OrderSide
from src.trading.brokers.paper import paper_broker
from src.trading.brokers.paper.paper_broker import PaperBroker
from src.utils.db._connection import _db_connection

_TODAY = date(2026, 3, 16)  # 月曜
_OPEN = 1000.0
_JST = ZoneInfo("Asia/Tokyo")


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setattr(paper_broker, "_today_jst", lambda: _TODAY)
    with _db_connection() as con:
        con.execute("DELETE FROM paper_positions")
        con.execute("DELETE FROM paper_orders")
        con.execute("UPDATE paper_balance SET balance = 1000000.0")


def _broker() -> PaperBroker:
    market_data = MagicMock(spec=MarketDataPort)
    # 今日の足（9:00 の始値を持つ）
    market_data.get_ohlcv.return_value = pd.DataFrame(
        {"Open": [_OPEN], "High": [_OPEN * 1.02], "Low": [_OPEN * 0.98], "Close": [_OPEN]},
        index=pd.to_datetime(["2026-03-16"]),
    )
    market_data.get_splits.return_value = pd.Series(
        [], index=pd.DatetimeIndex([]).tz_localize("Asia/Tokyo"), dtype=float
    )
    return PaperBroker(market_data_port=market_data, trade_diff_sink=InMemoryTradeDiffSink())


def _add_buy(order_id: str, created_at_utc: str, market: str | None = "jp") -> None:
    """成行の BUY を、created_at（列は tz 無し。DB のセッションは UTC）を決めて入れる"""
    with _db_connection() as con:
        con.execute(
            "INSERT INTO paper_orders (order_id, market, symbol, side, qty, price, "
            "order_type, status, created_at) "
            "VALUES (%s, %s, '7203', %s, 100, 0, 10, 'pending', %s)",
            [order_id, market, int(OrderSide.BUY), created_at_utc],
        )


def _status(order_id: str) -> str:
    with _db_connection() as con:
        return str(
            con.execute("SELECT status FROM paper_orders WHERE order_id=%s", [order_id]).fetchone()[
                0
            ]
        )


def test_created_at_reads_back_naive_utc():
    """send_order が書いた created_at は tz 無しで読み戻り、UTC の今の時刻である"""
    before = datetime.now(timezone.utc).replace(tzinfo=None)
    order_id = _broker().send_order("7203", OrderSide.BUY, 100)["order_id"]
    after = datetime.now(timezone.utc).replace(tzinfo=None)
    with _db_connection() as con:
        created_at = con.execute(
            "SELECT created_at FROM paper_orders WHERE order_id=%s", [order_id]
        ).fetchone()[0]
    assert created_at.tzinfo is None
    assert before - timedelta(minutes=1) <= created_at <= after + timedelta(minutes=1)


def test_order_after_close_of_bar_day_stays_pending(caplog):
    """今日の 16:30 JST に出た成行の BUY は、今日の足（始値）で約定させず pending に残す"""
    _add_buy("after-close", "2026-03-16 07:30:00")  # 16:30 JST
    with caplog.at_level(logging.WARNING):
        settled = _broker().settle_pending_orders()
    assert settled == []
    assert _status("after-close") == "pending"
    with _db_connection() as con:
        balance = float(con.execute("SELECT balance FROM paper_balance").fetchone()[0])
    assert balance == 1_000_000.0
    assert any(
        r.levelno == logging.WARNING and "引け" in r.getMessage() and "7203" in r.getMessage()
        for r in caplog.records
    )


def test_order_after_close_of_previous_day_fills_at_today_open():
    """前の営業日の 16:30 JST に出た注文は、今日の始値で約定する"""
    _add_buy("prev-day", "2026-03-13 07:30:00")  # 金曜 16:30 JST
    settled = _broker().settle_pending_orders()
    assert [s["order_id"] for s in settled] == ["prev-day"]
    assert settled[0]["fill_price"] == _OPEN


def test_order_before_open_of_today_fills_at_today_open():
    """今日の 8:50 JST に出た注文は、今日の始値で約定する"""
    _add_buy("pre-open", "2026-03-15 23:50:00")  # 03-16 08:50 JST（UTC では前の日）
    settled = _broker().settle_pending_orders()
    assert [s["order_id"] for s in settled] == ["pre-open"]
    assert settled[0]["fill_price"] == _OPEN


def test_us_order_is_not_cut_by_tse_close():
    """us の注文は東証の引けで区切らない"""
    _add_buy("us-order", "2026-03-16 07:30:00", market="us")
    settled = _broker().settle_pending_orders()
    assert [s["order_id"] for s in settled] == ["us-order"]


@pytest.mark.parametrize(
    "created_at, expected",
    [
        # tz 無しの値は UTC とみなす（DB のセッションは UTC）
        (datetime(2026, 3, 16, 6, 29), False),  # 15:29 JST
        (datetime(2026, 3, 16, 6, 30), True),  # 15:30 JST
        (datetime(2026, 3, 15, 23, 50), False),  # 03-16 08:50 JST
        # tz の有る値は JST に直す
        (datetime(2026, 3, 16, 16, 30, tzinfo=_JST), True),
        (datetime(2026, 3, 16, 8, 50, tzinfo=_JST), False),
        (datetime(2026, 3, 16, 7, 30, tzinfo=timezone.utc), True),  # 16:30 JST
        (datetime(2026, 3, 16, 5, 0, tzinfo=timezone.utc), False),  # 14:00 JST
        (None, False),
    ],
)
def test_created_after_bar_close_converts_to_jst(created_at, expected):
    assert paper_broker._created_after_bar_close(created_at, _TODAY) is expected
