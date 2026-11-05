"""16:30 の回が消えて流し直した日の、1日の新規買いの上限（#1660）

MAX_ORDERS_PER_RUN は回の中だけで数えていたので、消えた回が k 本の BUY を出していると、
流し直しがさらに上限まで出せた。その日の 16:30 の回が出した約定待ちの BUY を始めの値にする。
"""

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.domain.ports import MarketDataPort
from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.brokers.paper.paper_broker import PaperBroker
from src.trading.execution import selection
from src.trading.rule_execution import execute_rule_paper_trades
from src.utils.db._connection import _db_connection

_TODAY = date(2026, 10, 29)
# created_at の列は tz 無しの UTC
_TODAY_1635_JST = datetime(2026, 10, 29, 7, 35)
_TODAY_1529_JST = datetime(2026, 10, 29, 6, 29)
_TODAY_0850_JST = datetime(2026, 10, 28, 23, 50)
_YESTERDAY_1635_JST = datetime(2026, 10, 28, 7, 35)
_CAP = 5


class _ProcessKilled(BaseException):
    """プロセスが消えた回の代わり。銘柄ごとの except Exception では受けられない"""


def _buy_signals(count: int) -> list[dict]:
    return [{"symbol": str(3000 + i), "signal": 1, "price": 1000.0} for i in range(count)]


def _run(broker: BrokerBase, signals: list[dict]) -> tuple[dict[str, int], dict[str, int]]:
    """発注を1回流し、戻り値と判定の記録の skip_reasons を返す"""
    risk = MagicMock()
    risk.evaluate_trading_gate.return_value = MagicMock(is_allowed=True, reason=None)
    risk.calc_position_size.return_value = 100
    with (
        patch("src.trading.rule_execution.RiskManager", return_value=risk),
        patch(
            "src.trading.rule_execution.evaluate_correlation_gate",
            return_value=MagicMock(is_allowed=True, reason=None),
        ),
        patch("src.trading.rule_execution.get_symbol_sector", side_effect=lambda _, s: s),
        patch(
            "src.trading.rule_execution.get_ticker",
            side_effect=lambda market, symbol: f"{symbol}.T",
        ),
        patch("src.trading.rule_execution.MAX_ORDERS_PER_RUN", _CAP),
        patch("src.trading.rule_execution.MAX_POSITIONS", 20),
        patch("src.trading.rule_execution.save_rule_gate_record") as save,
    ):
        result = execute_rule_paper_trades(
            signals, "jp", trade_diff_sink=InMemoryTradeDiffSink(), broker=broker, record_gate=True
        )
    return result, dict(save.call_args.args[0].skip_reasons)


class TestDailyRuleBuyCapOnRerun:
    """紙の Broker と DB の約定待ちのまま、消えた回と流し直しを通す"""

    @pytest.fixture(autouse=True)
    def _clean_tables(self):
        self._clear()
        with patch(
            "src.trading.brokers.paper.paper_broker._today_jst",
            return_value=_TODAY,
        ):
            yield
        self._clear()

    @staticmethod
    def _clear() -> None:
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")
            con.execute("DELETE FROM paper_positions")
            con.execute("DELETE FROM paper_short_positions")

    @staticmethod
    def _broker() -> PaperBroker:
        market_data = MagicMock(spec=MarketDataPort)
        market_data.get_ohlcv.return_value = pd.DataFrame()
        return PaperBroker(market_data_port=market_data, trade_diff_sink=InMemoryTradeDiffSink())

    @staticmethod
    def _stamp_orders(created_at: datetime, predicted_at: str | None = None) -> None:
        """今ある注文の出た時刻を差し替える（テストが流れる時刻によらないように）"""
        with _db_connection() as con:
            con.execute(
                "UPDATE paper_orders SET created_at=%s, predicted_at=%s",
                [created_at, predicted_at],
            )

    @staticmethod
    def _pending_buy_count() -> int:
        with _db_connection() as con:
            row = con.execute(
                "SELECT COUNT(*) FROM paper_orders WHERE status='pending' AND side=%s",
                [int(OrderSide.BUY)],
            ).fetchone()
        return int(row[0])

    def _killed_run(self, signals: list[dict], placed: int) -> None:
        """BUY を placed 本出した後で消える回を流し、出た時刻を今日の 16:35 にする"""
        broker = self._broker()
        real_send = broker.send_order
        sent: list[str] = []

        def _send_then_die(symbol, *args, **kwargs):
            if len(sent) == placed:
                raise _ProcessKilled
            sent.append(symbol)
            return real_send(symbol, *args, **kwargs)

        with patch.object(broker, "send_order", side_effect=_send_then_die):
            with pytest.raises(_ProcessKilled):
                _run(broker, signals)
        self._stamp_orders(_TODAY_1635_JST)

    def _place_buys(self, symbols: list[str], created_at: datetime, predicted_at=None) -> None:
        broker = self._broker()
        for symbol in symbols:
            broker.send_order(symbol, OrderSide.BUY, 100)
        self._stamp_orders(created_at, predicted_at)

    @pytest.mark.parametrize("killed_buys", [1, 2, 4])
    def test_rerun_places_only_what_is_left_of_the_cap(self, killed_buys):
        """消えた回が k 本出した後の流し直しの新しい BUY は 上限 - k 本までで、1日で上限に収まること"""
        signals = _buy_signals(10)
        self._killed_run(signals, placed=killed_buys)

        result, skip_reasons = _run(self._broker(), signals)

        assert result["buy_orders"] == _CAP - killed_buys
        assert self._pending_buy_count() == _CAP
        assert skip_reasons["max_orders_per_run"] == 10 - _CAP
        assert skip_reasons["already_held"] == killed_buys

    def test_rerun_after_cap_was_reached_still_places_sells(self):
        """消えた回が上限まで BUY を出していても、流し直しの SELL は出ること"""
        with _db_connection() as con:
            con.execute(
                "INSERT INTO paper_positions (symbol, qty, avg_price, updated_at) "
                "VALUES ('1000', 100, 1000.0, CURRENT_TIMESTAMP)"
            )
        signals = _buy_signals(6) + [{"symbol": "1000", "signal": -1, "price": 1000.0}]
        self._killed_run(signals, placed=_CAP)

        result, skip_reasons = _run(self._broker(), signals)

        assert result == {"buy_orders": 0, "sell_orders": 1, "skipped": 6, "errors": 0}
        assert skip_reasons["max_orders_per_run"] == 1

    @pytest.mark.parametrize(
        ("created_at", "predicted_at"),
        [
            pytest.param(_YESTERDAY_1635_JST, None, id="yesterday-rule-run"),
            pytest.param(_TODAY_0850_JST, "20261029_085000", id="morning-prediction-run"),
            pytest.param(_TODAY_0850_JST, None, id="morning-order-without-metadata"),
            pytest.param(_TODAY_1529_JST, None, id="before-close"),
            pytest.param(_TODAY_1635_JST, "20261029_163500", id="evening-prediction-order"),
        ],
    )
    def test_other_pending_buys_do_not_reduce_todays_buys(self, created_at, predicted_at):
        """その日の 16:30 の回の BUY でない約定待ちは、今日の新規買いを減らさないこと"""
        self._place_buys(["7001", "7002", "7003"], created_at, predicted_at)

        result, _ = _run(self._broker(), _buy_signals(_CAP))

        assert result["buy_orders"] == _CAP

    def test_reader_returns_only_rule_buys_of_today(self):
        """読む関数は、今日の引けの後に出た予想に紐づかない約定待ちの BUY だけを返すこと"""
        self._place_buys(["7001"], _YESTERDAY_1635_JST)
        broker = self._broker()
        broker.send_order("7002", OrderSide.BUY, 100)
        broker.send_order("7003", OrderSide.SELL, 100)
        cancelled = broker.send_order("7004", OrderSide.BUY, 100)
        broker.cancel_order(cancelled["order_id"])
        with _db_connection() as con:
            con.execute(
                "UPDATE paper_orders SET created_at=%s WHERE symbol <> '7001'",
                [_TODAY_1635_JST],
            )

        assert broker.get_rule_buy_symbols_placed_today() == {"7002"}


class TestRuleBuysPlacedTodayReader:
    def _broker(self, positions: list[dict], reader: MagicMock | None) -> MagicMock:
        broker = MagicMock(spec=BrokerBase)
        broker.broker_name = "mock"
        broker.get_positions.return_value = positions
        broker.send_order.return_value = {"order_id": "ord-1660", "status": "pending"}
        broker.get_pending_buy_symbols = MagicMock(return_value=set())
        if reader is not None:
            broker.get_rule_buy_symbols_placed_today = reader
        return broker

    def test_broker_without_reader_counts_from_zero(self):
        """読む関数が無い Broker は 0 本から数えること"""
        broker = self._broker([], None)

        assert selection._get_rule_buy_symbols_placed_today(broker) == set()
        result, _ = _run(broker, _buy_signals(_CAP))

        assert result["buy_orders"] == _CAP

    def test_reader_failure_places_held_sell_then_reraises(self):
        """読めない日は約定待ちの読みが落ちる回と同じく、新規を止め保有の SELL を出して失敗にする"""
        error = RuntimeError("db down")
        held = [{"symbol": "1000", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0}]
        broker = self._broker(held, MagicMock(side_effect=error))
        signals = _buy_signals(1) + [{"symbol": "1000", "signal": -1, "price": 1000.0}]

        with pytest.raises(RuntimeError) as caught:
            _run(broker, signals)

        assert caught.value is error
        broker.send_order.assert_called_once_with(
            symbol="1000.T", side=OrderSide.SELL, qty=100, order_type=OrderType.MARKET
        )
