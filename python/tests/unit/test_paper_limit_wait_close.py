"""ユニットテスト: 場中の足で指値に届かない注文を失効させず、引けた足で判定する（#2108）

yfinance の東証の日足は、9:25 の回では寄りから数分の値動きしか含まない。その足で指値に
届かないときは pending に残し、その日の引けた足（次の営業日の朝の回では前の日の足）の
高値・安値で約定か失効にする。実 Postgres（conftest の `_isolate_db` でロールバック分離）を使う。
"""

from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.domain.ports import MarketDataPort
from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.trading.brokers.base import OrderSide, OrderType
from src.trading.brokers.paper import paper_broker
from src.trading.brokers.paper.paper_broker import PaperBroker
from src.utils.db._connection import _db_connection

_JST = ZoneInfo("Asia/Tokyo")
_LIMIT = 950.0


@pytest.fixture(autouse=True)
def _reset():
    with _db_connection() as con:
        con.execute("DELETE FROM paper_positions")
        con.execute("DELETE FROM paper_orders")
        con.execute("UPDATE paper_balance SET balance = 1000000.0")


def _clock(monkeypatch, now: datetime) -> None:
    monkeypatch.setattr(paper_broker, "_today_jst", lambda: now.date())
    monkeypatch.setattr(paper_broker, "_now_jst", lambda: now, raising=False)


def _bars(rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    """(日付, 始値, 高値, 安値) の行から日足を作る"""
    return pd.DataFrame(
        {
            "Open": [r[1] for r in rows],
            "High": [r[2] for r in rows],
            "Low": [r[3] for r in rows],
            "Close": [r[1] for r in rows],
        },
        index=pd.to_datetime([r[0] for r in rows]),
    )


def _broker(hist: pd.DataFrame) -> PaperBroker:
    market_data = MagicMock(spec=MarketDataPort)
    market_data.get_ohlcv.return_value = hist
    market_data.get_splits.return_value = pd.Series(
        [], index=pd.DatetimeIndex([]).tz_localize("Asia/Tokyo"), dtype=float
    )
    return PaperBroker(market_data_port=market_data, trade_diff_sink=InMemoryTradeDiffSink())


def _add_limit_buy(order_id: str = "lim") -> None:
    """D の 8:50 JST に出た、前の営業日のシグナルの買いの指値"""
    with _db_connection() as con:
        con.execute(
            "INSERT INTO paper_orders (order_id, market, predicted_at, symbol, side, qty, "
            "price, signal_price, order_type, status, created_at) "
            "VALUES (%s, 'jp', '20260313_200000', '7203', %s, 100, %s, 960.0, %s, "
            "'pending', '2026-03-15 23:50:00')",
            [order_id, int(OrderSide.BUY), _LIMIT, int(OrderType.LIMIT)],
        )


def _status(order_id: str = "lim") -> str:
    with _db_connection() as con:
        return str(
            con.execute("SELECT status FROM paper_orders WHERE order_id=%s", [order_id]).fetchone()[
                0
            ]
        )


def test_intraday_bar_not_reached_stays_pending(monkeypatch):
    """9:25 の場中の足で安値が買いの指値に届かないときは、失効させず pending に残す"""
    _clock(monkeypatch, datetime(2026, 3, 16, 9, 25, tzinfo=_JST))
    _add_limit_buy()
    hist = _bars([("2026-03-13", 990.0, 1000.0, 980.0), ("2026-03-16", 1000.0, 1005.0, 995.0)])
    settled = _broker(hist).settle_pending_orders()
    assert settled == []
    assert _status() == "pending"


def test_intraday_bar_reached_fills_as_before(monkeypatch):
    """場中の足で届いた指値は、今までどおり min(指値, 始値) で約定する"""
    _clock(monkeypatch, datetime(2026, 3, 16, 9, 25, tzinfo=_JST))
    _add_limit_buy()
    hist = _bars([("2026-03-13", 990.0, 1000.0, 980.0), ("2026-03-16", 1000.0, 1005.0, 940.0)])
    settled = _broker(hist).settle_pending_orders()
    assert [s["fill_price"] for s in settled] == [_LIMIT]
    assert _status() == "filled"


def test_closed_bar_same_day_judges(monkeypatch):
    """引けた後の回では、その日の足で届かなければ失効する"""
    _clock(monkeypatch, datetime(2026, 3, 16, 15, 45, tzinfo=_JST))
    _add_limit_buy()
    hist = _bars([("2026-03-13", 990.0, 1000.0, 980.0), ("2026-03-16", 1000.0, 1005.0, 995.0)])
    assert _broker(hist).settle_pending_orders() == []
    assert _status() == "expired"


@pytest.mark.parametrize(
    "next_rows",
    [
        [],  # 9:05 の回: 今日の行がまだ無い
        [("2026-03-17", 900.0, 910.0, 890.0)],  # 9:25 の回: 今日の場中の行がある
    ],
)
def test_next_morning_fills_on_previous_closed_bar(monkeypatch, next_rows):
    """据え置いた注文は、次の営業日の朝の回で前の日の引けた足が届いていれば、その足で約定する"""
    _clock(monkeypatch, datetime(2026, 3, 16, 9, 25, tzinfo=_JST))
    _add_limit_buy()
    intraday = _bars([("2026-03-13", 990.0, 1000.0, 980.0), ("2026-03-16", 1000.0, 1005.0, 995.0)])
    _broker(intraday).settle_pending_orders()
    assert _status() == "pending"

    _clock(monkeypatch, datetime(2026, 3, 17, 9, 5, tzinfo=_JST))
    closed = _bars([("2026-03-16", 1000.0, 1010.0, 945.0)] + next_rows)
    settled = _broker(closed).settle_pending_orders()
    # D の始値 1000 と指値 950 の小さい方。D+1 の安い始値 900 では約定させない
    assert [s["fill_price"] for s in settled] == [_LIMIT]
    assert _status() == "filled"


@pytest.mark.parametrize(
    "next_rows",
    [[], [("2026-03-17", 900.0, 910.0, 890.0)]],
)
def test_next_morning_expires_when_previous_closed_bar_not_reached(monkeypatch, next_rows):
    """前の日の引けた足でも届かなければ、次の日の足では約定させず失効する"""
    _clock(monkeypatch, datetime(2026, 3, 17, 9, 5, tzinfo=_JST))
    _add_limit_buy()
    closed = _bars([("2026-03-16", 1000.0, 1010.0, 960.0)] + next_rows)
    assert _broker(closed).settle_pending_orders() == []
    assert _status() == "expired"


def test_market_order_is_not_judged_on_previous_bar(monkeypatch):
    """成行は今までどおり、今日の行が無ければ前の日の足で約定させない"""
    _clock(monkeypatch, datetime(2026, 3, 17, 9, 5, tzinfo=_JST))
    with _db_connection() as con:
        con.execute(
            "INSERT INTO paper_orders (order_id, market, predicted_at, symbol, side, qty, "
            "price, order_type, status, created_at) "
            "VALUES ('mkt', 'jp', '20260313_200000', '7203', %s, 100, 0, %s, "
            "'pending', '2026-03-15 23:50:00')",
            [int(OrderSide.BUY), int(OrderType.MARKET)],
        )
    closed = _bars([("2026-03-16", 1000.0, 1010.0, 945.0)])
    assert _broker(closed).settle_pending_orders() == []
    assert _status("mkt") == "pending"
