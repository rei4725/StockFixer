"""16:30 のルールの発注の、失敗の回の補完（#1676）

失敗で終わった回を、間を空けて同じ日に1回だけ流し直す。流し直しは、失敗の回が出した約定待ちの
BUY と SELL を飛ばし、落ちた注文と判定できなかった分だけを出す。消えた回と違い、失敗の回は
銘柄ごとの例外を受けて最後まで走るので、本物のジョブと紙の Broker を失敗の回の形で通して確かめる。
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.domain.ports import MarketDataPort
from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.orchestration.job_errors import NonRetryableJobError
from src.orchestration.jobs import daily
from src.orchestration.scheduler_queue import SchedulerQueueManager
from src.rule_engine.types import RuleSignalBatch
from src.trading.brokers.base import OrderSide
from src.trading.brokers.paper.paper_broker import PaperBroker
from src.utils.db._connection import _db_connection
from src.utils.japan_time import isoformat_utc

_JOB_ID = "daily_rule_signals"
_PERIOD = "2026-10-29"  # 木曜
_GAP = timedelta(minutes=240)
_BUY, _SELL = int(OrderSide.BUY), int(OrderSide.SELL)


def _manager(func: Any, state_path: Path) -> SchedulerQueueManager:
    """本番の設定のまま、ジョブの関数だけを差し替えたマネージャを返す"""
    from run_scheduler import SCHEDULE_CONFIG

    return SchedulerQueueManager(
        {_JOB_ID: {**SCHEDULE_CONFIG[_JOB_ID], "func": func}}, state_file_path=str(state_path)
    )


def _failed_at(manager: SchedulerQueueManager) -> datetime:
    """定時の 10 分の遅れより後に終わった回。間の判定だけを見る"""
    return datetime(2026, 10, 29, 16, 41, tzinfo=manager.tz)


def _run_at(manager: SchedulerQueueManager, at: datetime, reason: str) -> None:
    with (
        patch("src.orchestration.scheduler_queue.ping_heartbeat"),
        patch.object(manager, "now", return_value=at),
    ):
        manager.run_job(_JOB_ID, reason=reason)


def _fail_at(manager: SchedulerQueueManager, at: datetime, reason: str = "scheduled") -> None:
    with pytest.raises(RuntimeError):
        _run_at(manager, at, reason)


def test_failed_run_waits_for_the_gap_then_recovers(tmp_path):
    """本番の設定で、失敗の回の5分後と間の1分前は補完せず、間を過ぎると補完すること"""
    manager = _manager(MagicMock(side_effect=RuntimeError("障害")), tmp_path / "state.json")
    failed = _failed_at(manager)
    _fail_at(manager, failed)

    assert manager.should_recover(_JOB_ID, now=failed + timedelta(minutes=5)) is False
    assert manager.should_recover(_JOB_ID, now=failed + _GAP - timedelta(minutes=1)) is False
    assert manager.should_recover(_JOB_ID, now=failed + _GAP) is True


def test_failed_run_is_called_once_within_the_run(tmp_path):
    """失敗の回は同じ回の中では呼び直さないこと（#836・#1075）"""
    func = MagicMock(side_effect=RuntimeError("障害"))
    manager = _manager(func, tmp_path / "state.json")
    _fail_at(manager, _failed_at(manager))

    assert func.call_count == 1


def test_failed_recovery_is_not_recovered_again(tmp_path):
    """補完の回も失敗したら、その日はもう補完しないこと"""
    func = MagicMock(side_effect=RuntimeError("障害"))
    manager = _manager(func, tmp_path / "state.json")
    failed = _failed_at(manager)
    _fail_at(manager, failed)

    recovery_at = failed + _GAP
    assert manager.should_recover(_JOB_ID, now=recovery_at) is True
    _fail_at(manager, recovery_at, reason="recovery")

    assert func.call_count == 2
    assert manager.should_recover(_JOB_ID, now=recovery_at + _GAP) is False
    last_of_the_day = datetime(2026, 10, 29, 23, 59, tzinfo=manager.tz)
    assert manager.should_recover(_JOB_ID, now=last_of_the_day) is False


def test_successful_recovery_is_not_recovered_again(tmp_path):
    """補完の回が成功したら、その日はもう動かないこと"""
    func = MagicMock(side_effect=[RuntimeError("障害"), None])
    manager = _manager(func, tmp_path / "state.json")
    failed = _failed_at(manager)
    _fail_at(manager, failed)
    recovery_at = failed + _GAP
    assert manager.should_recover(_JOB_ID, now=recovery_at) is True
    _run_at(manager, recovery_at, "recovery")

    assert manager.should_recover(_JOB_ID, now=recovery_at + _GAP) is False


def test_failed_run_is_not_recovered_on_the_next_day(tmp_path):
    """失敗の回も、次の日へ補完しないこと（#825）"""
    manager = _manager(MagicMock(side_effect=RuntimeError("障害")), tmp_path / "state.json")
    _fail_at(manager, _failed_at(manager))

    next_day = datetime(2026, 10, 30, 8, 0, tzinfo=manager.tz)
    assert manager.should_recover(_JOB_ID, now=next_day) is False
    assert manager.get_missed_past_periods(_JOB_ID, now=next_day) == []
    # 次の日の定時の後も、前の日の失敗は次の日の回数に入らず、次の日の補完の間も数えない
    after_next_schedule = datetime(2026, 10, 30, 16, 41, tzinfo=manager.tz)
    assert manager.should_recover(_JOB_ID, now=after_next_schedule) is True
    assert manager.get_missed_past_periods(_JOB_ID, now=after_next_schedule) == []


def test_lost_run_after_failed_run_is_recovered_without_a_new_gap(tmp_path):
    """失敗の回の補完が消えても、消えた回の後は間を待たず、失敗がもう1回で止まること"""
    state_path = tmp_path / "state.json"
    func = MagicMock(side_effect=RuntimeError("障害"))
    manager = _manager(func, state_path)
    failed = _failed_at(manager)
    _fail_at(manager, failed)

    lost_at = failed + _GAP
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["events"].append(
        {
            "job_id": _JOB_ID,
            "reason": "recovery",
            "status": "started",
            "period_key": _PERIOD,
            "run_id": "lost-recovery",
            "boot_id": "previous-boot",
            "started_at": isoformat_utc(lost_at),
        }
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")

    restarted = _manager(func, state_path)
    next_poll = lost_at + timedelta(minutes=5)
    assert restarted.should_recover(_JOB_ID, now=next_poll) is True
    _fail_at(restarted, next_poll, reason="recovery")
    assert restarted.should_recover(_JOB_ID, now=next_poll + _GAP) is False


class TestRecoveryAfterFailedOrders:
    """本物の 16:30 のジョブを、紙の Broker と DB の約定待ちのまま失敗の回の形で通す"""

    @pytest.fixture(autouse=True)
    def _clean_tables(self):
        self._clear()
        with _db_connection() as con:
            for symbol in ("1000", "2000"):
                con.execute(
                    "INSERT INTO paper_positions (symbol, qty, avg_price, updated_at) "
                    "VALUES (%s, 100, 1000.0, CURRENT_TIMESTAMP)",
                    [symbol],
                )
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

    @classmethod
    def _broker_failing_on(cls, *failing: str) -> PaperBroker:
        """銘柄を指定して注文を例外で落とす Broker。ほかの銘柄の注文は通る"""
        broker = cls._broker()
        real_send = broker.send_order

        def _send(symbol: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
            if symbol.replace(".T", "") in failing:
                raise RuntimeError(f"注文の失敗: {symbol}")
            return real_send(symbol, *args, **kwargs)

        broker.send_order = _send  # type: ignore[method-assign]
        return broker

    @staticmethod
    def _batch(*signals: tuple[str, int]) -> RuleSignalBatch:
        rows = [
            {"symbol": symbol, "rule": "rsi_contrarian", "signal": signal, "price": 1000.0}
            for symbol, signal in signals
        ]
        return RuleSignalBatch(signals=rows, unavailable=[], total=len(rows))

    @staticmethod
    def _orders() -> list[tuple[str, int]]:
        with _db_connection() as con:
            rows = con.execute(
                "SELECT symbol, side FROM paper_orders WHERE status='pending' ORDER BY symbol, side"
            ).fetchall()
        return [(str(symbol), int(side)) for symbol, side in rows]

    @staticmethod
    def _stamp_orders_after_the_close() -> None:
        """失敗の回の注文の時刻を、その日の 16:41 JST（tz 無しの列は UTC）にそろえる"""
        with _db_connection() as con:
            con.execute("UPDATE paper_orders SET created_at=%s", ["2026-10-29 07:41:00"])

    @contextmanager
    def _job_stage(self, brokers: list[PaperBroker], batch: RuleSignalBatch) -> Iterator[MagicMock]:
        """本物のジョブの外側だけを差し替え、Discord の日次の知らせのモックを渡す"""
        risk = MagicMock()
        risk.evaluate_trading_gate.return_value = MagicMock(is_allowed=True, reason=None)
        risk.calc_position_size.return_value = 100
        send_signals = MagicMock()
        with (
            patch("src.orchestration.jobs.daily._skip_if_jp_market_closed", return_value=False),
            patch("src.orchestration.jobs.daily.log_rule_signal_versions_stage"),
            patch("src.rule_engine.pipeline.generate_rule_signals", return_value=batch),
            patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter"),
            patch("src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink"),
            patch("src.infrastructure.discord_notification_adapter.DiscordNotificationAdapter"),
            patch("src.trading.brokers.account.factory.make_order_broker", side_effect=brokers),
            patch("src.reporting.discord.discord_utils.send_rule_daily_signals", send_signals),
            patch("src.scoring.rule_report.load_rule_track_record", return_value=None),
            patch("src.trading.rule_execution.save_rule_gate_record"),
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
            patch(
                "src.trading.brokers.paper.paper_broker._today_jst",
                return_value=date(2026, 10, 29),
            ),
        ):
            yield send_signals

    def _fail_then_recover(self, state_path: Path) -> None:
        """失敗の回を流し、間が明けた後の補完の回を流す"""
        manager = _manager(daily.run_daily_rule_signals, state_path)
        failed = _failed_at(manager)
        _fail_at(manager, failed)
        self._stamp_orders_after_the_close()

        assert manager.should_recover(_JOB_ID, now=failed + _GAP) is True
        _run_at(manager, failed + _GAP, "recovery")

    def test_recovery_places_only_the_orders_that_failed(self, tmp_path):
        """失敗の回の BUY と SELL に2本目を出さず、落ちた BUY と SELL だけを出すこと"""
        batch = self._batch(("3000", 1), ("1000", -1), ("4000", 1), ("2000", -1))
        brokers = [self._broker_failing_on("4000", "2000"), self._broker()]

        with self._job_stage(brokers, batch) as send_signals:
            self._fail_then_recover(tmp_path / "state.json")

        assert self._orders() == [
            ("1000", _SELL),
            ("2000", _SELL),
            ("3000", _BUY),
            ("4000", _BUY),
        ]
        # 失敗の回も知らせの後に失敗になるので、日次の知らせは2通になる
        assert send_signals.call_count == 2
        recovery_notice = send_signals.call_args_list[1].kwargs
        assert (recovery_notice["buy_orders"], recovery_notice["sell_orders"]) == (1, 1)

    def test_new_buys_of_the_day_stay_within_the_cap_after_recovery(self, tmp_path):
        """失敗の回の買いが上限まで出ていたら、補完は新規の買いを足さず、落ちた SELL だけ出すこと"""
        batch = self._batch(
            ("3000", 1), ("4000", 1), ("5000", 1), ("6000", 1), ("1000", -1), ("2000", -1)
        )
        brokers = [self._broker_failing_on("4000", "2000"), self._broker()]

        with (
            self._job_stage(brokers, batch),
            patch("src.trading.rule_execution.MAX_ORDERS_PER_RUN", 2),
        ):
            self._fail_then_recover(tmp_path / "state.json")

        assert self._orders() == [
            ("1000", _SELL),
            ("2000", _SELL),
            ("3000", _BUY),
            ("5000", _BUY),
        ]

    def test_recovery_fills_the_cap_left_by_the_failed_run(self, tmp_path):
        """失敗の回の買いが上限の前で落ちたら、補完は上限までだけ足すこと"""
        batch = self._batch(("3000", 1), ("4000", 1), ("5000", 1), ("6000", 1))
        brokers = [self._broker_failing_on("4000", "5000", "6000"), self._broker()]

        with (
            self._job_stage(brokers, batch),
            patch("src.trading.rule_execution.MAX_ORDERS_PER_RUN", 2),
        ):
            self._fail_then_recover(tmp_path / "state.json")

        assert self._orders() == [("3000", _BUY), ("4000", _BUY)]

    def test_recovery_after_the_new_buy_step_failed_adds_the_buys_only(self, tmp_path):
        """新規買いの手順が落ちて SELL だけ出した回の補完は、BUY だけを足すこと"""
        batch = self._batch(("3000", 1), ("1000", -1))
        failing = self._broker()
        failing.get_pending_buy_symbols = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("読めない")
        )

        with self._job_stage([failing, self._broker()], batch) as send_signals:
            self._fail_then_recover(tmp_path / "state.json")

        assert self._orders() == [("1000", _SELL), ("3000", _BUY)]
        # この形の失敗の回は知らせの前に落ちるので、知らせは補完の回の1通だけになる
        assert send_signals.call_count == 1

    def test_no_effective_rules_day_notifies_twice_and_stops(self, tmp_path):
        """有効ルールが0銘柄の日は、補完でも0銘柄のまま知らせが1通増え、そこで止まること"""
        batch = RuleSignalBatch(signals=[], unavailable=[], total=0)
        brokers = [self._broker(), self._broker()]
        manager = _manager(daily.run_daily_rule_signals, tmp_path / "state.json")
        failed = _failed_at(manager)

        with self._job_stage(brokers, batch) as send_signals:
            with pytest.raises(NonRetryableJobError):
                _run_at(manager, failed, "scheduled")
            assert manager.should_recover(_JOB_ID, now=failed + _GAP) is True
            with pytest.raises(NonRetryableJobError):
                _run_at(manager, failed + _GAP, "recovery")

        assert send_signals.call_count == 2
        assert manager.should_recover(_JOB_ID, now=failed + _GAP * 2) is False
        assert self._orders() == []
