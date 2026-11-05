"""
ユニットテスト: ペーパーの保有の読みの、1行の例外の受け方（#1580）

1銘柄の行の処理が例外を投げても、一覧は落とさず、その銘柄は評価できない行として返す。
最初の SELECT の失敗は今までどおり投げる（読めない日を0件にしない、#1451・#1458）。
"""

import logging
import math
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.domain.ports import MarketDataPort
from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.trading.brokers.base import OrderSide
from src.trading.brokers.paper import paper_broker, split_adjust
from src.trading.brokers.paper.paper_broker import PaperBroker
from src.trading.execution import _check_sl_tp_exits
from src.utils.db._connection import _db_connection

_TODAY = date(2026, 3, 15)
_BROKEN = "7203"
_HEALTHY = "6758"
_ROW_LOGGER = "src.trading.brokers.paper.position_rows"


@pytest.fixture(autouse=True)
def _fixed_today(monkeypatch):
    monkeypatch.setattr(paper_broker, "_today_jst", lambda: _TODAY)


@pytest.fixture(autouse=True)
def _empty_tables():
    with _db_connection() as con:
        con.execute("DELETE FROM paper_orders")
        con.execute("DELETE FROM paper_positions")
        con.execute("DELETE FROM paper_short_positions")


def _broker(broken_close: bool = False) -> PaperBroker:
    """broken_close なら _BROKEN の足に Close の列が無い（_latest_close の中で落ちる）"""
    good = pd.DataFrame(
        {"Open": [1000.0], "Close": [1020.0]},
        index=pd.to_datetime([_TODAY.isoformat()]),
    )

    def ohlcv(ticker: str, _period: str) -> pd.DataFrame:
        if broken_close and ticker == f"{_BROKEN}.T":
            return good.drop(columns=["Close"])
        return good

    market_data = MagicMock(spec=MarketDataPort)
    market_data.get_ohlcv.side_effect = ohlcv
    market_data.get_splits.return_value = pd.Series(dtype=float)
    return PaperBroker(market_data_port=market_data, trade_diff_sink=InMemoryTradeDiffSink())


def _fail_split_adjust(name: str):
    """split_adjust の name を、_BROKEN のときだけ例外を投げるものに差し替える"""
    original = getattr(split_adjust, name)

    def adjust(market_data, sym, *args):
        if sym == _BROKEN:
            raise RuntimeError("分割の書き戻しに失敗")
        return original(market_data, sym, *args)

    return patch.object(split_adjust, name, side_effect=adjust)


def _insert_long(symbol: str, qty: int, avg: float, updated_at: str = "2026-03-10") -> None:
    with _db_connection() as con:
        con.execute(
            "INSERT INTO paper_positions (symbol, qty, avg_price, updated_at) "
            "VALUES (%s, %s, %s, %s)",
            [symbol, qty, avg, updated_at],
        )


def _insert_short(symbol: str, qty: int, avg: float, updated_at: str = "2026-03-10") -> None:
    with _db_connection() as con:
        con.execute(
            "INSERT INTO paper_short_positions (symbol, qty, avg_short_price, updated_at) "
            "VALUES (%s, %s, %s, %s)",
            [symbol, qty, avg, updated_at],
        )


def _by_symbol(positions: list[dict]) -> dict[str, dict]:
    return {p["symbol"]: p for p in positions}


def _assert_row_failure_logged(caplog) -> None:
    errors = [r for r in caplog.records if r.levelname == "ERROR" and r.name == _ROW_LOGGER]
    assert len(errors) == 1
    assert _BROKEN in errors[0].getMessage()


def _assert_long_isolated(positions: list[dict]) -> None:
    assert len(positions) == 2
    rows = _by_symbol(positions)
    broken, healthy = rows[_BROKEN], rows[_HEALTHY]
    assert broken["qty"] == 100
    assert broken["avg_price"] == pytest.approx(1000.0)
    assert math.isnan(broken["current_price"])
    assert math.isnan(broken["unrealized_pnl"])
    assert healthy["qty"] == 200
    assert healthy["current_price"] == pytest.approx(1020.0)
    assert healthy["unrealized_pnl"] == pytest.approx((1020.0 - 1000.0) * 200)


def _assert_short_isolated(positions: list[dict]) -> None:
    assert len(positions) == 2
    rows = _by_symbol(positions)
    broken, healthy = rows[_BROKEN], rows[_HEALTHY]
    assert broken["qty"] == 100
    assert broken["pending_qty"] == broken["qty"]  # 分からない分は返済に出さない（#1225）
    assert broken["avg_short_price"] == pytest.approx(1200.0)
    assert math.isnan(broken["current_price"])
    assert math.isnan(broken["unrealized_pnl"])
    assert healthy["qty"] == 200
    assert healthy["pending_qty"] == 0
    assert healthy["current_price"] == pytest.approx(1020.0)
    assert healthy["unrealized_pnl"] == pytest.approx((1200.0 - 1020.0) * 200)


class TestGetPositionsRowIsolation:
    def _insert(self) -> None:
        _insert_long(_BROKEN, 100, 1000.0)
        _insert_long(_HEALTHY, 200, 1000.0)

    def test_close_read_failure_keeps_other_rows(self, caplog):
        self._insert()
        with caplog.at_level(logging.ERROR):
            positions = _broker(broken_close=True).get_positions()
        _assert_long_isolated(positions)
        _assert_row_failure_logged(caplog)

    def test_split_adjust_failure_keeps_other_rows(self, caplog):
        self._insert()
        with _fail_split_adjust("adjust_position_for_splits"), caplog.at_level(logging.ERROR):
            positions = _broker().get_positions()
        _assert_long_isolated(positions)
        _assert_row_failure_logged(caplog)

    def test_failure_after_split_adjust_returns_adjusted_qty(self):
        """直して書き戻した後に落ちた行は、書き戻した値（DB の今の値）で返す"""
        _insert_long(_BROKEN, 100, 1000.0, updated_at="2026-01-05")
        broker = _broker(broken_close=True)
        broker._market_data.get_splits.return_value = pd.Series(
            [2.0], index=pd.to_datetime(["2026-02-02"])
        )
        pos = broker.get_positions()[0]
        assert pos["qty"] == 200
        assert pos["avg_price"] == pytest.approx(500.0)
        assert math.isnan(pos["current_price"])
        with _db_connection() as con:
            row = con.execute(
                "SELECT qty FROM paper_positions WHERE symbol=%s", [_BROKEN]
            ).fetchone()
        assert row[0] == 200


class TestGetShortPositionsRowIsolation:
    def _insert(self) -> None:
        _insert_short(_BROKEN, 100, 1200.0)
        _insert_short(_HEALTHY, 200, 1200.0)

    def test_close_read_failure_keeps_other_rows(self, caplog):
        self._insert()
        with caplog.at_level(logging.ERROR):
            positions = _broker(broken_close=True).get_short_positions()
        _assert_short_isolated(positions)
        _assert_row_failure_logged(caplog)

    def test_split_adjust_failure_keeps_other_rows(self, caplog):
        self._insert()
        with _fail_split_adjust("adjust_short_position_for_splits"), caplog.at_level(logging.ERROR):
            positions = _broker().get_short_positions()
        _assert_short_isolated(positions)
        _assert_row_failure_logged(caplog)

    def test_failed_row_with_pending_short_is_not_coverable(self):
        """約定待ちを持つ行でも、落ちたら行の株数の全部を約定待ちとして返す"""
        _insert_short(_BROKEN, 150, 1200.0)
        with _db_connection() as con:
            con.execute(
                "INSERT INTO paper_orders (order_id, market, symbol, side, qty, price, "
                "order_type, status, created_at) "
                "VALUES ('o1', 'jp', %s, %s, 50, 0, 1, 'pending', CURRENT_TIMESTAMP)",
                [_BROKEN, int(OrderSide.SHORT)],
            )
        pos = _broker(broken_close=True).get_short_positions()[0]
        assert pos["qty"] == 150
        assert pos["pending_qty"] == 150


class TestFirstSelectStillRaises:
    """読めない日を0件にしない（#1451・#1458・#1465）"""

    @pytest.mark.parametrize("method", ["get_positions", "get_short_positions"])
    def test_select_failure_raises(self, method):
        cm = MagicMock()
        cm.__enter__.return_value.execute.side_effect = RuntimeError("SELECT に失敗")
        with patch.object(paper_broker, "_db_connection", return_value=cm):
            with pytest.raises(RuntimeError, match="SELECT に失敗"):
                getattr(_broker(), method)()


class TestSlTpOnRowFailure:
    """1銘柄の例外の朝に、残りの保有の損切りが出る"""

    _PARAMS = {"stop_loss_pct": 0.05, "take_profit_pct": 0.10}

    def _run(self, broker: PaperBroker) -> tuple[set[str], dict]:
        stats: dict = {"errors": 0, "sell_orders": 0, "short_orders": 0, "total_turnover": 0.0}
        with (
            patch("src.trading.risk_manager.get_optimal_params", return_value=self._PARAMS),
            patch("src.trading.execution.sl_tp._record_order"),
        ):
            triggered = _check_sl_tp_exits(
                broker, "jp", "paper", None, stats, InMemoryTradeDiffSink()
            )
        return triggered, stats

    @staticmethod
    def _ordered(side: OrderSide) -> set[str]:
        with _db_connection() as con:
            rows = con.execute(
                "SELECT symbol FROM paper_orders WHERE side=%s AND status='pending'", [int(side)]
            ).fetchall()
        return {str(sym) for (sym,) in rows}

    def test_long_stop_fires_for_remaining_positions(self):
        _insert_long(_BROKEN, 100, 1200.0)
        _insert_long(_HEALTHY, 200, 1200.0)  # 1020 は買値から -15%
        triggered, stats = self._run(_broker(broken_close=True))
        assert triggered == {_HEALTHY}
        assert self._ordered(OrderSide.SELL) == {_HEALTHY}
        assert stats["sl_tp_unevaluable_symbols"] == [_BROKEN]

    def test_short_stop_fires_without_covering_failed_row(self):
        _insert_short(_BROKEN, 100, 900.0)
        _insert_short(_HEALTHY, 200, 900.0)  # 1020 は売値から +13%
        triggered, _ = self._run(_broker(broken_close=True))
        assert triggered == {_HEALTHY}
        assert self._ordered(OrderSide.SHORT_COVER) == {_HEALTHY}
