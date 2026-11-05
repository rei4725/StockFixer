"""レジームレバレッジの判定が、まだ届いていない足を飛ばさないこと（#1608）

週次は、引けたはずの足が無ければ判定せず投げて補完に乗せる。日次は、足が遅れた回の noop で
判定の時刻を進めず、遅れて届いた足を次の回が判定する。足が2営業日以上古い回は投げて知らせに
乗せる(#1620)。判定する足の並びの途中の営業日の足が欠けた回も、投げて知らせに乗せる(#1633)。
"""

from dataclasses import asdict, replace
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.trading.regime_leverage_strategy.service import (
    run_regime_leverage_daily_margin_check,
    run_regime_leverage_weekly_check,
)
from src.trading.regime_leverage_strategy.types import (
    RegimeLeverageDecision,
    RegimeLeverageSnapshot,
)

_SERVICE = "src.trading.regime_leverage_strategy.service"
_JST = ZoneInfo("Asia/Tokyo")
_THURSDAY = "2026-11-05"
_FRIDAY = "2026-11-06"
_SATURDAY_SCHEDULED = datetime(2026, 11, 7, 6, 30, tzinfo=_JST)
_SUNDAY_RECOVERY = datetime(2026, 11, 8, 12, 0, tzinfo=_JST)
# ドル円146円・損切りの値段65,000円なので、安値445ドル以下で損切りになる
_RATE = 146.0
_STOP_LOW = 440.0
# 維持率 0.20 を割る安値(マージンコールが損切りより先)
_MARGIN_CALL_LOW = 300.0
# ここで見るのは SPY の足の遅れなので、ドル円の足はどの回の時計よりも新しい日に固定する
# (SPY の足に合わせると、ドル円の遅れの確かめが先に投げる。#1621)
_FRESH_FX_BAR = pd.Timestamp("2026-12-31")


def _clock(moment: datetime) -> type[datetime]:
    """now() が moment を返す datetime。コンテナの時計は UTC なので、naive の now は UTC にする"""

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[no-untyped-def, override]
            if tz is not None:
                return moment.astimezone(tz)
            return moment.astimezone(timezone.utc).replace(tzinfo=None)

    return _Frozen


def _utc_naive(moment: datetime) -> datetime:
    """DB の executed_at と同じ、タイムゾーンの無い UTC にする(#942)"""
    return moment.astimezone(timezone.utc).replace(tzinfo=None)


def _holding(executed_at: datetime, reason: str = "daily_noop") -> RegimeLeverageSnapshot:
    return RegimeLeverageSnapshot(
        id=1,
        executed_at=executed_at,
        action="noop",
        reason=reason,
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


def _port(
    last_bar: str,
    closes: tuple = (500.0, 500.0),
    lows: dict | None = None,
    without: tuple = (),
) -> MagicMock:
    """last_bar の日までの日足を返す口。closes は最初と最後の終値、lows は日ごとの安値、
    without は足の無い日"""
    idx = pd.bdate_range(end=last_bar, periods=260)
    prices = pd.Series(np.linspace(closes[0], closes[1], 260), index=idx)
    low = prices - 2
    for day, value in (lows or {}).items():
        low.loc[pd.Timestamp(day)] = value
    port = MagicMock()
    port.get_stock_data.return_value = pd.DataFrame(
        {"Open": prices, "High": prices + 2, "Low": low, "Close": prices, "Volume": 1_000_000},
        index=idx,
    ).drop(index=[pd.Timestamp(day) for day in without])
    port.get_forex_data.return_value = pd.DataFrame({"Close": [_RATE]}, index=[_FRESH_FX_BAR])
    return port


def _run_weekly(port: MagicMock, moment: datetime, snapshot) -> tuple:
    """週次の回を動かし、結果と insert_snapshot のモックを返す"""
    with (
        patch(f"{_SERVICE}.datetime", _clock(moment)),
        patch(f"{_SERVICE}.get_latest_snapshot", return_value=snapshot),
        patch(f"{_SERVICE}.insert_snapshot") as mock_insert,
    ):
        decision = run_regime_leverage_weekly_check(port)
    return decision, mock_insert


class _DailyBook:
    """日次の回を続けて動かす。書いた行を次の回が読むように、最新の行を覚えておく"""

    def __init__(
        self, snapshot: RegimeLeverageSnapshot, last_daily_check: datetime | None = None
    ) -> None:
        self.latest = snapshot
        self.last_daily_check = last_daily_check
        self.inserted: list[RegimeLeverageDecision] = []

    def run(self, port: MagicMock, moment: datetime) -> RegimeLeverageDecision:
        def _insert(decision: RegimeLeverageDecision) -> None:
            self.inserted.append(decision)
            self.latest = RegimeLeverageSnapshot(
                id=self.latest.id + 1, executed_at=_utc_naive(moment), **asdict(decision)
            )

        with (
            patch(f"{_SERVICE}.datetime", _clock(moment)),
            patch(f"{_SERVICE}.get_latest_snapshot", side_effect=lambda: self.latest),
            patch(f"{_SERVICE}.get_last_daily_check_at", return_value=self.last_daily_check),
            patch(f"{_SERVICE}.insert_snapshot", side_effect=_insert),
        ):
            decision = run_regime_leverage_daily_margin_check(port)
        assert decision is not None
        return decision


@pytest.mark.parametrize("moment", [_SATURDAY_SCHEDULED, _SUNDAY_RECOVERY])
def test_weekly_raises_without_insert_when_friday_bar_is_missing(moment):
    """保有中に金曜の足が無ければ、木曜の終値でレジーム転換を判定せず投げること"""
    port = _port(_THURSDAY, closes=(600.0, 400.0))

    with (
        patch(f"{_SERVICE}.datetime", _clock(moment)),
        patch(f"{_SERVICE}.get_latest_snapshot", return_value=_holding(datetime(2026, 11, 5))),
        patch(f"{_SERVICE}.insert_snapshot") as mock_insert,
        pytest.raises(RuntimeError, match=f"last_bar={_THURSDAY}.*expected={_FRIDAY}"),
    ):
        run_regime_leverage_weekly_check(port)

    mock_insert.assert_not_called()


def test_weekly_unheld_raises_without_insert_when_friday_bar_is_missing():
    """未保有でも、木曜の終値で新規に建てず投げること"""
    port = _port(_THURSDAY, closes=(400.0, 500.0))

    with (
        patch(f"{_SERVICE}.datetime", _clock(_SATURDAY_SCHEDULED)),
        patch(f"{_SERVICE}.get_latest_snapshot", return_value=None),
        patch(f"{_SERVICE}.insert_snapshot") as mock_insert,
        pytest.raises(RuntimeError, match="last_bar="),
    ):
        run_regime_leverage_weekly_check(port)

    mock_insert.assert_not_called()


@pytest.mark.parametrize("moment", [_SATURDAY_SCHEDULED, _SUNDAY_RECOVERY])
def test_weekly_judges_when_friday_bar_exists(moment):
    """金曜の足があれば今までどおり判定すること"""
    snapshot = _holding(datetime(2026, 11, 5))

    decision, mock_insert = _run_weekly(_port(_FRIDAY, closes=(600.0, 400.0)), moment, snapshot)

    assert decision.reason == "regime_flip"
    assert decision.spy_price_usd == 400.0
    mock_insert.assert_called_once_with(decision)


@pytest.mark.parametrize(
    "last_bar, moment",
    [
        # 2026-04-03 は Good Friday、土曜は夏時間の 06:30 JST
        ("2026-04-02", datetime(2026, 4, 4, 6, 30, tzinfo=_JST)),
        # 2026-12-25 は金曜のクリスマス
        ("2026-12-24", datetime(2026, 12, 26, 6, 30, tzinfo=_JST)),
        # 2026-07-03 は独立記念日の振替(7/4 が土曜)
        ("2026-07-02", datetime(2026, 7, 4, 6, 30, tzinfo=_JST)),
    ],
)
def test_weekly_judges_thursday_bar_when_friday_is_a_market_holiday(last_bar, moment):
    """金曜が休場の週は、木曜の足がその週の最後の足なので失敗にせず判定すること"""
    snapshot = _holding(datetime(2026, 3, 1))

    decision, mock_insert = _run_weekly(_port(last_bar, closes=(400.0, 500.0)), moment, snapshot)

    assert decision.reason == "weekly_noop"
    assert decision.spy_price_usd == 500.0
    mock_insert.assert_called_once_with(decision)


def test_weekly_retry_judges_friday_bar_after_it_arrives():
    """定時の回で投げ、足が届いた補完の回がその足で判定すること"""
    snapshot = _holding(datetime(2026, 11, 5))

    with pytest.raises(RuntimeError):
        _run_weekly(_port(_THURSDAY, closes=(600.0, 400.0)), _SATURDAY_SCHEDULED, snapshot)
    decision, _ = _run_weekly(
        _port(_FRIDAY, closes=(600.0, 400.0)),
        datetime(2026, 11, 7, 7, 0, tzinfo=_JST),
        snapshot,
    )

    assert decision.reason == "regime_flip"


@pytest.mark.parametrize(
    "low, reason",
    [(_STOP_LOW, "initial_stop"), (_MARGIN_CALL_LOW, "margin_call")],
)
def test_daily_judges_monday_low_when_bar_arrives_a_run_late(low, reason):
    """火曜の回で月曜の足が届かず、水曜の回で届いたら、月曜の安値で判定すること(冬時間)"""
    # 月曜 11/2 07:30 JST の回で金曜 10/30 の足まで判定済み
    book = _DailyBook(_holding(datetime(2026, 11, 1, 22, 30)))

    tuesday = book.run(_port("2026-10-30"), datetime(2026, 11, 3, 7, 30, tzinfo=_JST))
    assert tuesday.reason == "daily_noop"

    wednesday = book.run(
        _port("2026-11-03", lows={"2026-11-02": low}),
        datetime(2026, 11, 4, 7, 30, tzinfo=_JST),
    )
    assert wednesday.action == "exit"
    assert wednesday.reason == reason
    assert wednesday.spy_price_usd == low
    assert book.inserted == [wednesday]


def test_daily_keeps_check_time_when_new_bar_judged_but_newer_bar_is_late():
    """月曜の回が落ち、火曜の回に金曜の足だけが届いた回でも、判定の時刻を進めないこと"""
    # 金曜 10/30 07:30 JST の回で木曜 10/29 の足まで判定済み
    book = _DailyBook(_holding(datetime(2026, 10, 29, 22, 30)))

    tuesday = book.run(_port("2026-10-30"), datetime(2026, 11, 3, 7, 30, tzinfo=_JST))
    assert tuesday.reason == "daily_noop"

    wednesday = book.run(
        _port("2026-11-03", lows={"2026-11-02": _STOP_LOW}),
        datetime(2026, 11, 4, 7, 30, tzinfo=_JST),
    )
    assert wednesday.reason == "initial_stop"
    assert wednesday.spy_price_usd == _STOP_LOW


def test_daily_exits_on_late_run_without_waiting_for_the_late_bar():
    """足が遅れた回でも、手元の足で決済に当たれば決済を書くこと"""
    book = _DailyBook(_holding(datetime(2026, 10, 29, 22, 30)))

    decision = book.run(
        _port("2026-10-30", lows={"2026-10-30": _STOP_LOW}),
        datetime(2026, 11, 3, 7, 30, tzinfo=_JST),
    )

    assert decision.reason == "initial_stop"
    assert book.inserted == [decision]


def test_daily_records_noop_after_market_holiday():
    """月曜が休場の週の火曜の回は、待つ足が無いので今までどおり行を書くこと"""
    # 2026-09-07 は Labor Day。月曜 9/7 07:30 JST の回で金曜 9/4 の足まで判定済み
    book = _DailyBook(_holding(datetime(2026, 9, 6, 22, 30)))

    decision = book.run(_port("2026-09-04"), datetime(2026, 9, 8, 7, 30, tzinfo=_JST))

    assert decision.reason == "daily_noop"
    assert book.inserted == [decision]


@pytest.mark.parametrize(
    "last_bar, moment, expected",
    [
        # 水曜の回で火曜 11/3 が引けたはずなのに、金曜 10/30 の足のまま
        ("2026-10-30", datetime(2026, 11, 4, 7, 30, tzinfo=_JST), "2026-11-03"),
        # 週末をまたぐ: 火曜の回で月曜 11/2 が引けたはずなのに、木曜 10/29 の足のまま
        ("2026-10-29", datetime(2026, 11, 3, 7, 30, tzinfo=_JST), "2026-11-02"),
        # 休場をまたぐ: Labor Day 9/7 を数えず、金曜 9/4 と火曜 9/8 の2営業日が足りない
        ("2026-09-03", datetime(2026, 9, 9, 7, 30, tzinfo=_JST), "2026-09-08"),
    ],
)
def test_daily_raises_without_insert_when_bars_are_two_business_days_old(
    last_bar, moment, expected
):
    """足が2営業日以上古い回は、noop の行を書かずに投げて知らせに乗せること(#1620)"""
    book = _DailyBook(_holding(datetime(2026, 8, 1, 22, 30)))

    with pytest.raises(RuntimeError, match=f"last_bar={last_bar}, expected={expected}"):
        book.run(_port(last_bar), moment)

    assert book.inserted == []


def test_daily_raises_when_only_already_checked_bars_remain():
    """足が止まり、前の回が判定した足しか残っていない回でも投げること(#1620)"""
    # 月曜 11/2 07:30 JST の回で金曜 10/30 の足まで判定済み
    book = _DailyBook(_holding(datetime(2026, 11, 1, 22, 30)))

    with pytest.raises(RuntimeError, match="last_bar=2026-10-30"):
        book.run(_port("2026-10-30"), datetime(2026, 11, 4, 7, 30, tzinfo=_JST))

    assert book.inserted == []


@pytest.mark.parametrize(
    "last_bar, moment",
    [
        # 火曜の回で月曜 11/2 の足だけが遅れている(週末は数えない)
        ("2026-10-30", datetime(2026, 11, 3, 7, 30, tzinfo=_JST)),
        # 水曜の回で火曜 9/8 の足だけが遅れている(Labor Day 9/7 は数えない)
        ("2026-09-04", datetime(2026, 9, 9, 7, 30, tzinfo=_JST)),
    ],
)
def test_daily_one_business_day_late_returns_noop_without_insert(last_bar, moment):
    """1営業日の遅れの回は、今までどおり投げず、行も書かないこと"""
    book = _DailyBook(_holding(datetime(2026, 8, 1, 22, 30)))

    decision = book.run(_port(last_bar), moment)

    assert decision.reason == "daily_noop"
    assert book.inserted == []


@pytest.mark.parametrize(
    "low, reason",
    [(_STOP_LOW, "initial_stop"), (_MARGIN_CALL_LOW, "margin_call")],
)
def test_daily_exits_on_hand_bars_even_when_bars_are_old(low, reason):
    """足が2営業日以上古くても、手元の足で決済に当たれば投げずに決済を書くこと"""
    book = _DailyBook(_holding(datetime(2026, 10, 28, 22, 30)))

    decision = book.run(
        _port("2026-10-30", lows={"2026-10-29": low}),
        datetime(2026, 11, 4, 7, 30, tzinfo=_JST),
    )

    assert decision.reason == reason
    assert book.inserted == [decision]


def test_daily_judges_stale_days_after_bars_arrive():
    """投げた回は判定の時刻を進めず、足が届いた回が止まっていた間の安値を判定すること"""
    # 月曜 11/2 07:30 JST の回で金曜 10/30 の足まで判定済み
    book = _DailyBook(_holding(datetime(2026, 11, 1, 22, 30)))

    with pytest.raises(RuntimeError):
        book.run(_port("2026-10-30"), datetime(2026, 11, 4, 7, 30, tzinfo=_JST))

    decision = book.run(
        _port("2026-11-04", lows={"2026-11-02": _STOP_LOW}),
        datetime(2026, 11, 5, 7, 30, tzinfo=_JST),
    )
    assert decision.reason == "initial_stop"
    assert decision.spy_price_usd == _STOP_LOW


def test_daily_unheld_returns_none_even_when_bars_are_old():
    """未保有の回は、足を取りに行かず今までどおり None を返すこと"""
    flat = replace(_holding(datetime(2026, 8, 1, 22, 30)), shares=0.0)
    port = _port("2026-10-30")

    with (
        patch(f"{_SERVICE}.datetime", _clock(datetime(2026, 11, 4, 7, 30, tzinfo=_JST))),
        patch(f"{_SERVICE}.get_latest_snapshot", return_value=flat),
        patch(f"{_SERVICE}.insert_snapshot") as mock_insert,
    ):
        assert run_regime_leverage_daily_margin_check(port) is None

    port.get_stock_data.assert_not_called()
    mock_insert.assert_not_called()


# 11/3(火)の足が欠けた表。水曜の足まで届き、木曜 07:30 JST の回で判定する
_GAP_DAY = "2026-11-03"
_GAP_LAST_BAR = "2026-11-04"
_GAP_MOMENT = datetime(2026, 11, 5, 7, 30, tzinfo=_JST)
# 火曜 11/3 07:30 JST の回で月曜 11/2 の足まで判定済み
_CHECKED_MONDAY = datetime(2026, 11, 2, 22, 30)
# 月曜 11/2 07:30 JST の回で金曜 10/30 の足まで判定済み
_CHECKED_FRIDAY = datetime(2026, 11, 1, 22, 30)


@pytest.mark.parametrize("checked_at", [_CHECKED_MONDAY, _CHECKED_FRIDAY])
def test_daily_raises_without_insert_when_middle_bar_is_missing(checked_at):
    """前の回と届いた足の間、または足と足の間の1営業日が欠けたら、行を書かず投げること(#1633)"""
    book = _DailyBook(_holding(checked_at))
    latest = book.latest

    with pytest.raises(RuntimeError, match=_GAP_DAY):
        book.run(_port(_GAP_LAST_BAR, without=(_GAP_DAY,)), _GAP_MOMENT)

    assert book.inserted == []
    assert book.latest is latest


def test_daily_raises_for_missing_bar_when_latest_row_is_weekly():
    """最新の行が週次の行でも、前の日次の回の後の欠けを見つけて投げること"""
    book = _DailyBook(_holding(datetime(2026, 11, 3), "weekly_noop"), _CHECKED_MONDAY)

    with pytest.raises(RuntimeError, match=_GAP_DAY):
        book.run(_port(_GAP_LAST_BAR, without=(_GAP_DAY,)), _GAP_MOMENT)

    assert book.inserted == []


def test_daily_raises_for_missing_bar_when_last_bar_is_one_day_late():
    """最後の足が1営業日遅れの回でも、途中の欠けがあれば投げること"""
    book = _DailyBook(_holding(_CHECKED_MONDAY))
    friday_morning = datetime(2026, 11, 6, 7, 30, tzinfo=_JST)

    with pytest.raises(RuntimeError, match=_GAP_DAY):
        book.run(_port(_GAP_LAST_BAR, without=(_GAP_DAY,)), friday_morning)

    assert book.inserted == []


@pytest.mark.parametrize(
    "low, reason",
    [(_STOP_LOW, "initial_stop"), (_MARGIN_CALL_LOW, "margin_call")],
)
def test_daily_judges_missing_day_low_after_bar_arrives(low, reason):
    """投げた回は判定の時刻を進めず、欠けた足が届いた回がその日の安値を判定すること"""
    book = _DailyBook(_holding(_CHECKED_MONDAY))

    with pytest.raises(RuntimeError):
        book.run(_port(_GAP_LAST_BAR, without=(_GAP_DAY,)), _GAP_MOMENT)

    decision = book.run(
        _port(_GAP_LAST_BAR, lows={_GAP_DAY: low}),
        datetime(2026, 11, 5, 7, 45, tzinfo=_JST),
    )
    assert decision.reason == reason
    assert decision.spy_price_usd == low


@pytest.mark.parametrize("low_day", ["2026-11-02", _GAP_LAST_BAR])
def test_daily_exits_on_hand_bars_even_when_middle_bar_is_missing(low_day):
    """途中の足が欠けていても、手元の足で決済に当たれば投げずに決済を書くこと"""
    book = _DailyBook(_holding(_CHECKED_FRIDAY))

    decision = book.run(
        _port(_GAP_LAST_BAR, lows={low_day: _STOP_LOW}, without=(_GAP_DAY,)),
        _GAP_MOMENT,
    )

    assert decision.reason == "initial_stop"
    assert book.inserted == [decision]


@pytest.mark.parametrize(
    "checked_at, last_bar, without, moment",
    [
        # 週末: 土曜 11/7 07:30 JST の回で金曜の足まで判定済み。月曜 11/9 の足が届いた
        (
            datetime(2026, 11, 6, 22, 30),
            "2026-11-09",
            (),
            datetime(2026, 11, 10, 7, 30, tzinfo=_JST),
        ),
        # Labor Day 9/7: 金曜 9/4 の足の次が火曜 9/8 の足
        (
            datetime(2026, 9, 4, 22, 30),
            "2026-09-08",
            ("2026-09-07",),
            datetime(2026, 9, 9, 7, 30, tzinfo=_JST),
        ),
        # Labor Day をまたぐ2本を同じ回で判定する
        (
            datetime(2026, 9, 3, 22, 30),
            "2026-09-08",
            ("2026-09-07",),
            datetime(2026, 9, 9, 7, 30, tzinfo=_JST),
        ),
        # 感謝祭 11/26: 水曜 11/25 の足の次が金曜 11/27 の足
        (
            datetime(2026, 11, 24, 22, 30),
            "2026-11-27",
            ("2026-11-26",),
            datetime(2026, 11, 28, 7, 30, tzinfo=_JST),
        ),
        # Good Friday 4/3 と週末: 木曜 4/2 の足の次が月曜 4/6 の足
        (
            datetime(2026, 4, 1, 22, 30),
            "2026-04-06",
            ("2026-04-03",),
            datetime(2026, 4, 7, 7, 30, tzinfo=_JST),
        ),
    ],
)
def test_daily_does_not_count_weekend_or_market_holiday_as_missing(
    checked_at, last_bar, without, moment
):
    """週末と米国の休場をまたぐ足の並びは欠けに数えず、今までどおり行を書くこと"""
    book = _DailyBook(_holding(checked_at))

    decision = book.run(_port(last_bar, without=without), moment)

    assert decision.reason == "daily_noop"
    assert book.inserted == [decision]
