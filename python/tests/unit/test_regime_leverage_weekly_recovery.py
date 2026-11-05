"""レジームレバレッジの週次判定の、同じ ISO 週の補完（#1544）"""

import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.orchestration.scheduler_queue import SchedulerQueueManager  # noqa: E402
from src.trading.regime_leverage_strategy.service import (  # noqa: E402
    run_regime_leverage_weekly_check,
)
from src.trading.regime_leverage_strategy.types import (  # noqa: E402
    RegimeLeverageDecision,
    RegimeLeverageSnapshot,
)

_JOB_ID = "regime_leverage_weekly"
# 2026-11-02 は月曜、11-07 は土曜、11-08 は日曜。どれも 2026-W45
_WEEK_KEY = "2026-W45"
_SATURDAY = 7
_SUNDAY = 8


def _error(index: int) -> dict:
    return {
        "job_id": _JOB_ID,
        "reason": "scheduled",
        "status": "error",
        "period_key": _WEEK_KEY,
        "run_id": f"run-{index}",
    }


def _success() -> dict:
    return {
        "job_id": _JOB_ID,
        "reason": "scheduled",
        "status": "success",
        "period_key": _WEEK_KEY,
        "run_id": "run-ok",
    }


def _manager(tmp_path, events: list[dict] | None = None) -> SchedulerQueueManager:
    from run_scheduler import SCHEDULE_CONFIG

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"events": events or []}), encoding="utf-8")
    return SchedulerQueueManager(SCHEDULE_CONFIG, state_file_path=str(state_path))


def _limit() -> int:
    from run_scheduler import SCHEDULE_CONFIG

    return SCHEDULE_CONFIG[_JOB_ID]["max_executions_per_period"]


def test_limit_leaves_room_after_failed_runs():
    """落ちた回の後も補完できる上限であること"""
    assert _limit() == 12


def test_error_on_saturday_is_recovered_after_delay(tmp_path):
    """土曜 06:30 の回が error で終わったら、補完の猶予の後に補完すること"""
    manager = _manager(tmp_path, [_error(0)])

    before_delay = datetime(2026, 11, _SATURDAY, 6, 59, tzinfo=manager.tz)
    after_delay = datetime(2026, 11, _SATURDAY, 7, 0, tzinfo=manager.tz)
    assert manager.should_recover(_JOB_ID, now=before_delay) is False
    assert manager.should_recover(_JOB_ID, now=after_delay) is True


@pytest.mark.parametrize("events_count", [0, 1])
def test_sunday_recovers_week_without_success(tmp_path, events_count):
    """土曜に動かなかった週も、土曜に落ちた週も、同じ ISO 週の日曜に補完すること"""
    manager = _manager(tmp_path, [_error(i) for i in range(events_count)])

    start = datetime(2026, 11, _SUNDAY, 0, 0, tzinfo=manager.tz)
    end = datetime(2026, 11, _SUNDAY, 23, 59, tzinfo=manager.tz)
    assert manager.should_recover(_JOB_ID, now=start) is True
    assert manager.should_recover(_JOB_ID, now=end) is True


@pytest.mark.parametrize("day", [_SATURDAY, _SUNDAY])
def test_week_with_success_is_not_recovered(tmp_path, day):
    """同じ週に成功があれば、土曜も日曜も補完しないこと"""
    manager = _manager(tmp_path, [_error(0), _success()])

    now = datetime(2026, 11, day, 12, 0, tzinfo=manager.tz)
    assert manager.should_recover(_JOB_ID, now=now) is False


@pytest.mark.parametrize("day", [_SATURDAY, _SUNDAY])
def test_week_at_limit_is_not_recovered(tmp_path, day):
    """回数が上限に届いたら補完せず、上限の1つ手前では補完すること"""
    now_args = (2026, 11, day, 12, 0)

    below = _manager(tmp_path, [_error(i) for i in range(_limit() - 1)])
    assert below.should_recover(_JOB_ID, now=datetime(*now_args, tzinfo=below.tz)) is True

    at_limit = _manager(tmp_path, [_error(i) for i in range(_limit())])
    assert at_limit.should_recover(_JOB_ID, now=datetime(*now_args, tzinfo=at_limit.tz)) is False


@pytest.mark.parametrize("day", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("events_count", [0, 1])
def test_weekdays_are_not_recovered(tmp_path, day, events_count):
    """同じ週の月〜金曜は、回の記録の有無によらず補完しないこと"""
    manager = _manager(tmp_path, [_error(i) for i in range(events_count)])

    now = datetime(2026, 11, day, 12, 0, tzinfo=manager.tz)
    assert manager.should_recover(_JOB_ID, now=now) is False


def test_next_monday_does_not_recover_previous_week(tmp_path):
    """次の週の月曜は ISO 週が変わり、前の週の失敗を補完しないこと"""
    manager = _manager(tmp_path, [_error(0)])

    now = datetime(2026, 11, 9, 12, 0, tzinfo=manager.tz)
    assert manager.should_recover(_JOB_ID, now=now) is False


def test_sunday_recovery_counts_in_same_week_and_stops_after_success(tmp_path):
    """日曜の補完の回が土曜の回と同じ週に数えられ、成功の後は止まること"""
    from run_scheduler import SCHEDULE_CONFIG

    calls = {"n": 0}

    def _succeed() -> None:
        calls["n"] += 1

    config = {_JOB_ID: {**SCHEDULE_CONFIG[_JOB_ID], "func": _succeed, "max_retries": 0}}
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"events": [_error(0)]}), encoding="utf-8")
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))

    now = datetime(2026, 11, _SUNDAY, 12, 0, tzinfo=manager.tz)
    manager.now = lambda: now  # type: ignore[method-assign]
    assert manager.should_recover(_JOB_ID, now=now) is True
    assert manager.run_job(_JOB_ID, reason="recovery") is True

    assert calls["n"] == 1
    assert manager.has_success(_JOB_ID, _WEEK_KEY) is True
    assert manager.should_recover(_JOB_ID, now=now) is False


class _SaturdayRun(datetime):
    """now() が土曜 11/7 06:30 JST(= UTC 11/6 21:30)を返す datetime"""

    @classmethod
    def now(cls, tz=None):  # type: ignore[no-untyped-def, override]
        moment = datetime(2026, 11, 6, 21, 30, tzinfo=timezone.utc)
        return moment.replace(tzinfo=None) if tz is None else moment.astimezone(tz)


def _market_port(start_price: float, end_price: float) -> MagicMock:
    """start_price から end_price まで一直線に動いた、金曜 11/6 で終わる SPY の足を返す口"""
    idx = pd.bdate_range(end="2026-11-06", periods=260)
    prices = pd.Series(np.linspace(start_price, end_price, 260), index=idx)
    port = MagicMock()
    port.get_stock_data.return_value = pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 2,
            "Low": prices - 2,
            "Close": prices,
            "Volume": 1_000_000,
        },
        index=idx,
    )
    port.get_forex_data.return_value = pd.DataFrame({"Close": [145.0]}, index=[idx[-1]])
    return port


def _run_twice(port: MagicMock, first: RegimeLeverageSnapshot | None) -> list:
    """同じ週に判定を2回動かす。1回目の記録を2回目が読むように、記録を覚えておく"""
    latest = {"snapshot": first}

    def _insert(decision: RegimeLeverageDecision) -> None:
        latest["snapshot"] = RegimeLeverageSnapshot(
            id=1, executed_at=datetime(2026, 11, 7, 6, 30), **asdict(decision)
        )

    base = "src.trading.regime_leverage_strategy.service"
    with (
        patch(f"{base}.datetime", _SaturdayRun),
        patch(f"{base}.get_latest_snapshot", side_effect=lambda: latest["snapshot"]),
        patch(f"{base}.insert_snapshot", side_effect=_insert),
    ):
        return [run_regime_leverage_weekly_check(port) for _ in range(2)]


def test_second_run_after_entry_does_not_add_to_position():
    """建てた後の2回目は noop で、株数と建値が1回目のままであること"""
    first, second = _run_twice(_market_port(400.0, 500.0), None)

    assert first.action == "entry"
    assert first.shares > 0
    assert second.action == "noop"
    assert second.shares == first.shares
    assert second.entry_price_jpy == first.entry_price_jpy
    assert second.equity_at_entry_jpy == first.equity_at_entry_jpy


def test_second_run_after_regime_flip_stays_flat():
    """レジーム転換で決済した後の2回目は noop で、建て直さず現金が変わらないこと"""
    holding = RegimeLeverageSnapshot(
        id=1,
        executed_at=datetime(2026, 10, 31, 6, 30),
        action="entry",
        reason="regime_entry",
        spy_price_usd=500.0,
        usdjpy_rate=145.0,
        shares=27.0,
        entry_date=datetime(2026, 10, 31),
        entry_price_jpy=72500.0,
        entry_commission_jpy=0.0,
        equity_at_entry_jpy=1_000_000.0,
        stop_price_jpy=None,
        equity_now_jpy=1_000_000.0,
        maintenance_ratio=None,
    )

    first, second = _run_twice(_market_port(500.0, 400.0), holding)

    assert first.action == "exit"
    assert first.reason == "regime_flip"
    assert second.action == "noop"
    assert second.shares == 0.0
    assert second.equity_now_jpy == first.equity_now_jpy
