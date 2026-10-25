"""ユニットテスト: ペーパーの約定の1件分をトランザクションで囲む（#1088）

約定の途中の文が落ちたら、残高・保有・注文の行を約定前のまま戻して pending に残す。
実 Postgres（conftest の `_isolate_db` でロールバック分離）を使う。テストの `_db_connection` は
with ごとに SAVEPOINT を切り、本番の自動確定と違って書きかけが戻ってしまうため、ここでは
SAVEPOINT を切らずに共有接続をそのまま渡す形に差し替える。
"""

from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Iterator
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.domain.ports import MarketDataPort
from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.trading.brokers.paper import paper_broker
from src.trading.brokers.paper.paper_broker import PaperBroker
from src.utils.db import _connection

_TODAY = date(2026, 3, 10)
_OPEN = 1000.0
_INSERT_POSITION = "INSERT INTO paper_positions"
_DELETE_POSITION = "DELETE FROM paper_positions"
_FILLED = "status='filled'"


class _FailingCon:
    """指定の SQL を含む文で例外を投げる、共有接続の包み"""

    def __init__(self, con: Any, fail_on: dict[str, str | None]) -> None:
        self._con = con
        self._fail_on = fail_on

    def execute(self, sql: str, params: Any = None) -> Any:
        target = self._fail_on.get("sql")
        if target and target in sql:
            raise RuntimeError(f"書き込みの失敗: {target}")
        return self._con.execute(sql, params)

    def transaction(self) -> Any:
        return self._con.transaction()


@pytest.fixture
def fail_on(monkeypatch) -> dict[str, str | None]:
    """paper_broker の接続を、自動確定の本番と同じく SAVEPOINT を切らない形にする"""
    state: dict[str, str | None] = {"sql": None}

    @contextmanager
    def _plain_connection(lock_timeout: float | None = None) -> Iterator[_FailingCon]:
        yield _FailingCon(_connection._test_connection, state)

    monkeypatch.setattr(paper_broker, "_db_connection", _plain_connection)
    monkeypatch.setattr(paper_broker, "_today_jst", lambda: _TODAY)
    return state


@pytest.fixture(autouse=True)
def _setup():
    with _connection._db_connection() as con:
        con.execute("DELETE FROM paper_positions")
        con.execute("DELETE FROM paper_orders")
        con.execute("UPDATE paper_balance SET balance = 1000000.0")


def _broker() -> PaperBroker:
    market_data = MagicMock(spec=MarketDataPort)
    market_data.get_ohlcv.return_value = pd.DataFrame(
        {"Open": [_OPEN], "High": [_OPEN * 1.02], "Low": [_OPEN * 0.98], "Close": [_OPEN]},
        index=pd.to_datetime(["2026-03-10"]),
    )
    market_data.get_splits.return_value = pd.Series(
        [], index=pd.DatetimeIndex([]).tz_localize("Asia/Tokyo"), dtype=float
    )
    return PaperBroker(market_data_port=market_data, trade_diff_sink=InMemoryTradeDiffSink())


def _add_position(qty: int = 100, avg: float = 900.0) -> None:
    with _connection._db_connection() as con:
        con.execute(
            "INSERT INTO paper_positions (symbol, qty, avg_price, updated_at) "
            "VALUES ('7203', %s, %s, '2026-03-02 01:00:00')",
            [qty, avg],
        )


def _add_order(order_id: str, side: int, qty: int = 100, predicted_at: str | None = None) -> None:
    with _connection._db_connection() as con:
        con.execute(
            "INSERT INTO paper_orders (order_id, market, predicted_at, symbol, side, qty, price, "
            "order_type, status, created_at) "
            "VALUES (%s, 'jp', %s, '7203', %s, %s, 0, 10, 'pending', '2026-03-09 07:30:00')",
            [order_id, predicted_at, side, qty],
        )


def _state(order_id: str) -> tuple:
    """残高・保有・注文の行"""
    with _connection._db_connection() as con:
        balance = float(con.execute("SELECT balance FROM paper_balance").fetchone()[0])
        position = con.execute(
            "SELECT qty, avg_price FROM paper_positions WHERE symbol='7203'"
        ).fetchone()
        order = con.execute(
            "SELECT status, fill_price, filled_at, realized_pnl FROM paper_orders "
            "WHERE order_id=%s",
            [order_id],
        ).fetchone()
    return balance, position, order


@pytest.mark.parametrize("sql", [_INSERT_POSITION, _FILLED])
def test_buy_failure_rolls_back_to_before_fill(fail_on, sql):
    _add_order("buy1", 1)
    before = _state("buy1")
    fail_on["sql"] = sql
    assert _broker().settle_pending_orders() == []
    assert _state("buy1") == before
    assert before[2][0] == "pending"


@pytest.mark.parametrize("sql", [_DELETE_POSITION, _FILLED])
def test_sell_failure_rolls_back_to_before_fill(fail_on, sql):
    _add_position()
    _add_order("sell1", 2)
    before = _state("sell1")
    fail_on["sql"] = sql
    assert _broker().settle_pending_orders() == []
    assert _state("sell1") == before
    assert before[2][0] == "pending"


def test_buy_retry_after_failure_fills_once(fail_on):
    _add_order("buy1", 1)
    broker = _broker()
    fail_on["sql"] = _FILLED
    broker.settle_pending_orders()
    fail_on["sql"] = None
    [fill] = broker.settle_pending_orders()
    balance, position, order = _state("buy1")
    assert fill["order_id"] == "buy1"
    assert balance == pytest.approx(1000000.0 - _OPEN * 100)
    assert position == (100, pytest.approx(_OPEN))
    assert order[0] == "filled"
    assert broker.last_settle_errors == 0


def test_sell_retry_after_failure_fills_once(fail_on):
    _add_position()
    _add_order("sell1", 2)
    broker = _broker()
    fail_on["sql"] = _DELETE_POSITION
    broker.settle_pending_orders()
    fail_on["sql"] = None
    [fill] = broker.settle_pending_orders()
    balance, position, order = _state("sell1")
    assert fill["order_id"] == "sell1"
    assert balance == pytest.approx(1000000.0 + _OPEN * 100)
    assert position is None
    assert order[0] == "filled"
    assert order[3] == pytest.approx((_OPEN - 900.0) * 100)


def test_error_count_is_separate_from_deferred(fail_on, caplog):
    """例外の件数は数え、当日の行が無い据え置きは数えない"""
    _add_order("buy1", 1)
    # シグナル日が当日なので、最後の行が当日でも「当日の行が無い」扱いで据え置く
    _add_order("buy_deferred", 1, predicted_at="20260310_163000")
    fail_on["sql"] = _INSERT_POSITION
    broker = _broker()
    with caplog.at_level("ERROR"):
        assert broker.settle_pending_orders() == []
    assert broker.last_settle_errors == 1
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert any("例外" in m and "1 件" in m for m in errors)
    assert _state("buy_deferred")[2][0] == "pending"


_JST = ZoneInfo("Asia/Tokyo")


def test_daily_settle_returns_error_count_and_notifies():
    """9:05 の約定は例外の件数を返し、完了とは別にエラーの知らせを出す"""
    from src.orchestration.jobs import daily

    with (
        patch.dict("os.environ", {}, clear=True),
        patch("src.trading.brokers.paper.paper_broker.PaperBroker") as mock_broker_cls,
        patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter"),
        patch("src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink"),
        patch("src.reporting.discord.discord_utils.send_daily_settle_completion") as mock_done,
        patch("src.reporting.discord.discord_utils.send_webhook_notification") as mock_notify,
    ):
        mock_broker_cls.return_value.settle_pending_orders.return_value = []
        mock_broker_cls.return_value.last_settle_errors = 2
        assert daily.run_daily_settle_orders() == 2
    mock_done.assert_called_once_with(settled_count=0)
    mock_notify.assert_called_once()
    assert "2 件" in mock_notify.call_args.args[1]


def test_daily_settle_without_errors_does_not_notify_error():
    from src.orchestration.jobs import daily

    with (
        patch.dict("os.environ", {}, clear=True),
        patch("src.trading.brokers.paper.paper_broker.PaperBroker") as mock_broker_cls,
        patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter"),
        patch("src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink"),
        patch("src.reporting.discord.discord_utils.send_daily_settle_completion"),
        patch("src.reporting.discord.discord_utils.send_webhook_notification") as mock_notify,
    ):
        mock_broker_cls.return_value.settle_pending_orders.return_value = [{"qty": 1}]
        mock_broker_cls.return_value.last_settle_errors = 0
        assert daily.run_daily_settle_orders() == 0
    mock_notify.assert_not_called()


def test_cutoff_message_separates_deferred_and_errors():
    """締めの知らせは、当日の行が無い件数と例外の件数を分けて出す"""
    from src.orchestration.jobs import settle_retry

    now = datetime(2026, 10, 22, 9, 55, tzinfo=_JST)
    with (
        patch.object(settle_retry, "count_pending_paper_orders", side_effect=[3, 3]),
        patch.object(settle_retry, "run_daily_settle_orders", return_value=1),
        patch("src.reporting.discord.discord_utils.send_webhook_notification") as mock_notify,
    ):
        settle_retry.run_daily_settle_retry(now=now)
    message = mock_notify.call_args.args[1]
    assert "当日の行が無い 2 件" in message
    assert "約定処理の例外 1 件" in message
