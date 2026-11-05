"""レジームレバレッジが止まった古いドル円で判定しないこと（#1621）

ドル円の最後の足が、引けたはずの SPY の足の日より2平日以上古ければ投げて、行を書かずに
スケジューラーのやり直しと補完と知らせに乗せる。1平日の遅れは今までどおり通す。
"""

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.trading.regime_leverage_strategy.service import (
    run_regime_leverage_daily_margin_check,
    run_regime_leverage_weekly_check,
)
from src.trading.regime_leverage_strategy.types import RegimeLeverageSnapshot

_SERVICE = "src.trading.regime_leverage_strategy.service"
_JST = ZoneInfo("Asia/Tokyo")
_RATE = 146.0
# 2026-11-07 は土曜。PR 1619 の後の最初の週次の回で、引けたはずの足は金曜 11/6
_FRIDAY = "2026-11-06"
_SATURDAY_RUN = datetime(2026, 11, 7, 6, 30, tzinfo=_JST)
_SUNDAY_RECOVERY = datetime(2026, 11, 8, 12, 0, tzinfo=_JST)
# 冬時間の火曜 07:30 JST の日次の回で、引けたはずの足は月曜 11/9
_MONDAY = "2026-11-09"
_TUESDAY_RUN = datetime(2026, 11, 10, 7, 30, tzinfo=_JST)


def _clock(moment: datetime) -> type[datetime]:
    """now() が moment を返す datetime。コンテナの時計は UTC なので、naive の now は UTC にする"""

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[no-untyped-def, override]
            if tz is not None:
                return moment.astimezone(tz)
            return moment.astimezone(timezone.utc).replace(tzinfo=None)

    return _Frozen


def _holding() -> RegimeLeverageSnapshot:
    return RegimeLeverageSnapshot(
        id=1,
        executed_at=datetime(2026, 11, 1),
        action="noop",
        reason="daily_noop",
        spy_price_usd=500.0,
        usdjpy_rate=145.0,
        shares=27.0,
        entry_date=datetime(2026, 6, 1),
        entry_price_jpy=72500.0,
        entry_commission_jpy=0.0,
        equity_at_entry_jpy=1_000_000.0,
        stop_price_jpy=65000.0,
        equity_now_jpy=1_000_000.0,
        maintenance_ratio=None,
    )


def _port(spy_last_bar: str, fx_last_bar: str) -> MagicMock:
    """SPY は spy_last_bar まで、ドル円は fx_last_bar までの足を返す口"""
    idx = pd.bdate_range(end=spy_last_bar, periods=260)
    prices = pd.Series(np.linspace(400.0, 500.0, 260), index=idx)
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
    fx_idx = pd.bdate_range(end=fx_last_bar, periods=3)
    port.get_forex_data.return_value = pd.DataFrame({"Close": [999.0, 999.0, _RATE]}, index=fx_idx)
    return port


def _run(
    check: Callable[[Any], Any], port: MagicMock, moment: datetime, snapshot: Any
) -> tuple[Any, MagicMock]:
    """判定の回を動かし、結果と insert_snapshot のモックを返す"""
    with (
        patch(f"{_SERVICE}.datetime", _clock(moment)),
        patch(f"{_SERVICE}.get_latest_snapshot", return_value=snapshot),
        patch(f"{_SERVICE}.get_last_daily_check_at", return_value=None),
        patch(f"{_SERVICE}.insert_snapshot") as mock_insert,
    ):
        return check(port), mock_insert


def _assert_raises_without_insert(
    check: Callable[[Any], Any],
    port: MagicMock,
    moment: datetime,
    snapshot: Any,
    fx_last_bar: str,
    expected: str,
) -> None:
    with (
        patch(f"{_SERVICE}.datetime", _clock(moment)),
        patch(f"{_SERVICE}.get_latest_snapshot", return_value=snapshot),
        patch(f"{_SERVICE}.get_last_daily_check_at", return_value=None),
        patch(f"{_SERVICE}.insert_snapshot") as mock_insert,
        pytest.raises(RuntimeError, match=f"last_fx_bar={fx_last_bar}.*expected={expected}"),
    ):
        check(port)

    mock_insert.assert_not_called()


def test_daily_raises_without_insert_when_rate_is_two_weekdays_old():
    """建玉があり、ドル円が2平日古ければ、日次の判定が投げて行を書かないこと"""
    fx_last_bar = "2026-11-05"

    _assert_raises_without_insert(
        run_regime_leverage_daily_margin_check,
        _port(_MONDAY, fx_last_bar),
        _TUESDAY_RUN,
        _holding(),
        fx_last_bar,
        _MONDAY,
    )


@pytest.mark.parametrize("snapshot", [None, _holding()], ids=["unheld", "holding"])
def test_weekly_raises_without_insert_when_rate_is_two_weekdays_old(snapshot):
    """週次の判定は、未保有でも保有中でも、ドル円が2平日古ければ投げて行を書かないこと"""
    fx_last_bar = "2026-11-04"

    _assert_raises_without_insert(
        run_regime_leverage_weekly_check,
        _port(_FRIDAY, fx_last_bar),
        _SATURDAY_RUN,
        snapshot,
        fx_last_bar,
        _FRIDAY,
    )


def test_daily_raises_when_rate_stopped_days_ago_inside_the_window():
    """取得の窓の中に行が残っていても、止まったレートでは判定しないこと"""
    fx_last_bar = "2026-11-02"

    _assert_raises_without_insert(
        run_regime_leverage_daily_margin_check,
        _port(_MONDAY, fx_last_bar),
        _TUESDAY_RUN,
        _holding(),
        fx_last_bar,
        _MONDAY,
    )


@pytest.mark.parametrize(
    "fx_last_bar",
    [_MONDAY, "2026-11-06"],
    ids=["same_day", "one_weekday_old_over_the_weekend"],
)
def test_daily_judges_and_writes_when_rate_is_fresh_enough(fx_last_bar):
    """同じ日と1平日古いレートは、今までどおり判定して行を書くこと"""
    decision, mock_insert = _run(
        run_regime_leverage_daily_margin_check,
        _port(_MONDAY, fx_last_bar),
        _TUESDAY_RUN,
        _holding(),
    )

    assert decision.reason == "daily_noop"
    assert decision.usdjpy_rate == _RATE
    mock_insert.assert_called_once_with(decision)


@pytest.mark.parametrize("snapshot", [None, _holding()], ids=["unheld", "holding"])
@pytest.mark.parametrize("moment", [_SATURDAY_RUN, _SUNDAY_RECOVERY], ids=["saturday", "sunday"])
@pytest.mark.parametrize(
    "fx_last_bar", [_FRIDAY, "2026-11-05"], ids=["same_day", "one_weekday_old"]
)
def test_weekly_does_not_count_weekend_as_delay(fx_last_bar, moment, snapshot):
    """土曜の定時の回も日曜の補完の回も、週末の日数を遅れに数えず判定して行を書くこと"""
    decision, mock_insert = _run(
        run_regime_leverage_weekly_check, _port(_FRIDAY, fx_last_bar), moment, snapshot
    )

    assert decision.usdjpy_rate == _RATE
    mock_insert.assert_called_once_with(decision)


def test_monday_morning_run_does_not_count_weekend_as_delay():
    """月曜の朝の回は、引けたはずの足が金曜なので、金曜のレートで判定すること"""
    decision, mock_insert = _run(
        run_regime_leverage_daily_margin_check,
        _port(_FRIDAY, _FRIDAY),
        datetime(2026, 11, 9, 7, 30, tzinfo=_JST),
        _holding(),
    )

    assert decision.reason == "daily_noop"
    mock_insert.assert_called_once_with(decision)


@pytest.mark.parametrize(
    "spy_last_bar, fx_last_bar, moment",
    [
        # 2026-04-03 は Good Friday、土曜の週次の回で引けたはずの足は木曜
        ("2026-04-02", "2026-04-02", datetime(2026, 4, 4, 6, 30, tzinfo=_JST)),
        # 同じ回に、ドル円が1平日古い
        ("2026-04-02", "2026-04-01", datetime(2026, 4, 4, 6, 30, tzinfo=_JST)),
        # 2026-12-25 は金曜のクリスマスで、ドル円も足が出ない年がある
        ("2026-12-24", "2026-12-24", datetime(2026, 12, 26, 6, 30, tzinfo=_JST)),
        # 2027-01-01 は金曜の元日で、ドル円も足が出ない年がある
        ("2026-12-31", "2026-12-31", datetime(2027, 1, 2, 6, 30, tzinfo=_JST)),
    ],
    ids=["good_friday", "good_friday_one_weekday_old", "christmas", "new_year"],
)
def test_weekly_does_not_raise_across_market_holiday(spy_last_bar, fx_last_bar, moment):
    """休場をまたぐ回で、引けたはずの足の日より古くないレートなら投げないこと"""
    decision, mock_insert = _run(
        run_regime_leverage_weekly_check,
        _port(spy_last_bar, fx_last_bar),
        moment,
        _holding(),
    )

    assert decision.usdjpy_rate == _RATE
    mock_insert.assert_called_once_with(decision)


@pytest.mark.parametrize(
    "fx_last_bar", ["2026-09-04", "2026-09-03"], ids=["same_day", "one_weekday_old"]
)
def test_daily_does_not_raise_after_monday_market_holiday(fx_last_bar):
    """月曜の休場の翌朝は、引けたはずの足が金曜なので、1平日の遅れまで通すこと"""
    # 2026-09-07 は Labor Day
    decision, mock_insert = _run(
        run_regime_leverage_daily_margin_check,
        _port("2026-09-04", fx_last_bar),
        datetime(2026, 9, 8, 7, 30, tzinfo=_JST),
        _holding(),
    )

    assert decision.reason == "daily_noop"
    mock_insert.assert_called_once_with(decision)


def test_us_holiday_counts_as_weekday_because_rate_moves_that_day():
    """ドル円は米国の休場の日も動くので、その日も遅れに数えて投げること"""
    fx_last_bar = "2026-09-04"
    expected = "2026-09-08"

    _assert_raises_without_insert(
        run_regime_leverage_daily_margin_check,
        _port(expected, fx_last_bar),
        datetime(2026, 9, 9, 7, 30, tzinfo=_JST),
        _holding(),
        fx_last_bar,
        expected,
    )


def test_fx_only_holiday_on_a_weekday_passes_as_one_weekday_delay():
    """米国が開いていてドル円の足だけが無い平日は、1平日の遅れとして通すこと"""
    decision, mock_insert = _run(
        run_regime_leverage_weekly_check,
        _port(_FRIDAY, "2026-11-05"),
        _SATURDAY_RUN,
        _holding(),
    )

    assert decision.usdjpy_rate == _RATE
    mock_insert.assert_called_once_with(decision)


def test_rate_newer_than_expected_bar_is_not_stale():
    """ドル円の足が引けたはずの足より新しくても投げないこと(週末の足・引け前の足)"""
    decision, mock_insert = _run(
        run_regime_leverage_daily_margin_check,
        _port(_MONDAY, "2026-11-10"),
        _TUESDAY_RUN,
        _holding(),
    )

    assert decision.reason == "daily_noop"
    mock_insert.assert_called_once_with(decision)


def test_daily_retry_judges_after_rate_catches_up():
    """止まったレートで投げた後、レートが届いた回が判定して行を書くこと"""
    stale = _port(_MONDAY, "2026-11-05")
    with pytest.raises(RuntimeError, match="last_fx_bar="):
        _run(run_regime_leverage_daily_margin_check, stale, _TUESDAY_RUN, _holding())

    decision, mock_insert = _run(
        run_regime_leverage_daily_margin_check,
        _port(_MONDAY, _MONDAY),
        datetime(2026, 11, 10, 8, 0, tzinfo=_JST),
        _holding(),
    )

    assert decision.reason == "daily_noop"
    mock_insert.assert_called_once_with(decision)
