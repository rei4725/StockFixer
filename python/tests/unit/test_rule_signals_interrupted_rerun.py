"""16:30 のルールの発注の、前の起動で消えた回の補完（#1652）

デプロイの起き直しで消えた回を同じ日のうちに1回だけ流し直す。流し直しは、消えた回が出した
約定待ちの BUY と SELL を飛ばし、まだ出していない注文だけを出す。
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.domain.ports import MarketDataPort  # noqa: E402
from src.infrastructure.in_memory import InMemoryTradeDiffSink  # noqa: E402
from src.orchestration.scheduler_queue import SchedulerQueueManager  # noqa: E402
from src.trading.brokers.base import OrderSide  # noqa: E402
from src.trading.brokers.paper.paper_broker import PaperBroker  # noqa: E402
from src.trading.rule_execution import execute_rule_paper_trades  # noqa: E402
from src.utils.db._connection import _db_connection  # noqa: E402

_JOB_ID = "daily_rule_signals"
_PERIOD = "2026-10-29"  # 木曜


class _ProcessKilled(BaseException):
    """プロセスが消えた回の代わり。銘柄ごとの except Exception では受けられない"""


def _now(manager: SchedulerQueueManager) -> datetime:
    """16:30 + 10分を過ぎた時刻"""
    return datetime(2026, 10, 29, 16, 41, tzinfo=manager.tz)


def _config(func) -> dict:
    from run_scheduler import SCHEDULE_CONFIG

    return {_JOB_ID: {**SCHEDULE_CONFIG[_JOB_ID], "func": func}}


def _write_started(state_path: Path) -> None:
    """前の起動のプロセスが始めの記録だけを残して消えた状態を作る"""
    event = {
        "job_id": _JOB_ID,
        "reason": "scheduled",
        "status": "started",
        "period_key": _PERIOD,
        "run_id": "run-a",
        "boot_id": "previous-boot",
        "started_at": "2020-01-01T00:00:00+00:00",
    }
    state_path.write_text(json.dumps({"events": [event]}), encoding="utf-8")


def test_interrupted_run_is_recovered_on_the_same_day(tmp_path):
    """前の起動で消えた回を、同じ日のうちに補完の対象にすること"""
    from run_scheduler import SCHEDULE_CONFIG

    state_path = tmp_path / "state.json"
    _write_started(state_path)
    manager = SchedulerQueueManager(SCHEDULE_CONFIG, state_file_path=str(state_path))

    assert manager.should_recover(_JOB_ID, now=_now(manager)) is True
    assert manager.get_execution_count(_JOB_ID, _PERIOD) == 0


def test_interrupted_run_is_not_recovered_on_the_next_day(tmp_path):
    """補完は同じ日のうちだけで、次の日へ持ち越さないこと（#825）"""
    from run_scheduler import SCHEDULE_CONFIG

    state_path = tmp_path / "state.json"
    _write_started(state_path)
    manager = SchedulerQueueManager(SCHEDULE_CONFIG, state_file_path=str(state_path))

    next_morning = datetime(2026, 10, 30, 8, 0, tzinfo=manager.tz)
    assert manager.should_recover(_JOB_ID, now=next_morning) is False
    assert manager.get_missed_past_periods(_JOB_ID, now=next_morning) == []


def test_recovery_stops_after_two_consecutive_interruptions(tmp_path):
    """消えた回の後の補完は1回までで、2回続けて消えたら止まること（C-014）"""
    state_path = tmp_path / "state.json"
    calls = {"n": 0}
    snapshot: dict[str, bytes] = {}

    def _killed_func():
        calls["n"] += 1
        snapshot["state"] = state_path.read_bytes()

    config = _config(_killed_func)
    first = SchedulerQueueManager(config, state_file_path=str(state_path))
    now = _now(first)
    first.now = lambda: now  # type: ignore[method-assign]
    first.run_job(_JOB_ID, reason="scheduled")
    state_path.write_bytes(snapshot["state"])

    for _ in range(10):
        # 起き直したプロセスとして状態ファイルを読み直す
        manager = SchedulerQueueManager(config, state_file_path=str(state_path))
        manager.now = lambda: now  # type: ignore[method-assign]
        if not manager.should_recover(_JOB_ID, now=now):
            break
        manager.run_job(_JOB_ID, reason="recovery")
        state_path.write_bytes(snapshot["state"])

    assert calls["n"] == 2


def test_failed_run_is_not_called_again(tmp_path):
    """例外で失敗した回は、同じ回の中でも補完でも呼び直さないこと（#836・#1075）"""
    state_path = tmp_path / "state.json"
    func = MagicMock(side_effect=RuntimeError("注文の失敗"))
    manager = SchedulerQueueManager(_config(func), state_file_path=str(state_path))
    now = _now(manager)
    manager.now = lambda: now  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        manager.run_job(_JOB_ID, reason="scheduled")

    assert func.call_count == 1
    restarted = SchedulerQueueManager(_config(func), state_file_path=str(state_path))
    assert restarted.should_recover(_JOB_ID, now=now) is False
    assert restarted.get_execution_count(_JOB_ID, _PERIOD) == 1


class TestRerunOfInterruptedOrders:
    """発注の途中で消えた回を、紙の Broker と DB の約定待ちのまま流し直す"""

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

    @staticmethod
    def _run(broker: PaperBroker, signals: list[dict]) -> dict[str, int]:
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
        ):
            return execute_rule_paper_trades(
                signals, "jp", trade_diff_sink=InMemoryTradeDiffSink(), broker=broker
            )

    @staticmethod
    def _pending_orders() -> list[tuple[str, int]]:
        with _db_connection() as con:
            rows = con.execute(
                "SELECT symbol, side FROM paper_orders WHERE status='pending' ORDER BY symbol, side"
            ).fetchall()
        return [(str(symbol), int(side)) for symbol, side in rows]

    def test_rerun_places_only_orders_the_killed_run_did_not_place(self):
        """消えた回の BUY と SELL に2本目を出さず、残りの BUY と SELL だけを出すこと"""
        signals = [
            {"symbol": "3000", "signal": 1, "price": 1000.0},
            {"symbol": "1000", "signal": -1, "price": 1000.0},
            {"symbol": "4000", "signal": 1, "price": 1000.0},
            {"symbol": "2000", "signal": -1, "price": 1000.0},
        ]
        killed_broker = self._broker()
        real_send = killed_broker.send_order
        sent: list[str] = []

        def _send_then_die(symbol, *args, **kwargs):
            if len(sent) == 2:
                raise _ProcessKilled
            sent.append(symbol)
            return real_send(symbol, *args, **kwargs)

        with patch.object(killed_broker, "send_order", side_effect=_send_then_die):
            with pytest.raises(_ProcessKilled):
                self._run(killed_broker, signals)
        assert sent == ["3000.T", "1000.T"]

        result = self._run(self._broker(), signals)

        buy, sell = int(OrderSide.BUY), int(OrderSide.SELL)
        assert self._pending_orders() == [
            ("1000", sell),
            ("2000", sell),
            ("3000", buy),
            ("4000", buy),
        ]
        assert result == {"buy_orders": 1, "sell_orders": 1, "skipped": 2, "errors": 0}

    def test_rerun_after_all_orders_were_placed_adds_nothing(self):
        """全部の注文を出した後で消えた回の流し直しは、注文を1本も足さないこと"""
        signals = [
            {"symbol": "3000", "signal": 1, "price": 1000.0},
            {"symbol": "1000", "signal": -1, "price": 1000.0},
        ]
        self._run(self._broker(), signals)
        placed = self._pending_orders()

        result = self._run(self._broker(), signals)

        assert self._pending_orders() == placed
        assert result == {"buy_orders": 0, "sell_orders": 0, "skipped": 2, "errors": 0}
