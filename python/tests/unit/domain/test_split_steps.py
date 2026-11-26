"""domain の split_steps の直接のテスト（#2185）。

段差の線 0.55・1.8、比の近さ 0.05、5営業日の境、時差の付いた index の日付を、
浮動小数のちょうどの境を避けて手前と先で確かめる。期待値の線は本番の定数を import せず
ここに数で書く。線の値を変えた差分を拾うためである。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from src.domain.split_steps import find_price_steps, is_pending_split, pending_split_steps


def _frame(
    closes: List[object], start: str = "2023-03-01", tz: Optional[str] = None
) -> pd.DataFrame:
    index = pd.bdate_range(start, periods=len(closes), tz=tz)
    return pd.DataFrame({"Close": pd.Series(closes, index=index, dtype=object)})


def _day(text: str) -> np.datetime64:
    return np.datetime64(text, "D")


# --- find_price_steps -------------------------------------------------------


def test_find_price_steps_none_empty_and_no_close_column() -> None:
    assert find_price_steps(None) == []
    assert find_price_steps(pd.DataFrame()) == []
    no_close = pd.DataFrame({"Open": [100.0, 50.0]}, index=pd.bdate_range("2023-03-01", periods=2))
    assert find_price_steps(no_close) == []


@pytest.mark.parametrize("next_close", [55.0, 180.0])
def test_find_price_steps_exact_lines_are_not_steps(next_close: float) -> None:
    # 前日比がちょうど 0.55 と 1.8 は段差でない
    assert find_price_steps(_frame([100.0, next_close])) == []


@pytest.mark.parametrize("next_close", [54.9, 180.1])
def test_find_price_steps_just_past_lines_are_steps(next_close: float) -> None:
    steps = find_price_steps(_frame([100.0, next_close]))
    assert [day for day, _ in steps] == ["2023-03-02"]
    assert steps[0][1] == pytest.approx(next_close / 100.0)


def test_find_price_steps_skips_nan_and_non_numeric_closes() -> None:
    # 2・3本目の足は飛ばし、1本目と4本目で比を取る
    steps = find_price_steps(_frame([100.0, np.nan, "abc", 50.0]))
    assert steps == [("2023-03-06", 0.5)]


def test_find_price_steps_nan_between_equal_closes_is_not_a_step() -> None:
    assert find_price_steps(_frame([100.0, np.nan, 100.0])) == []


@pytest.mark.parametrize("hour", ["00:00", "19:00"])
@pytest.mark.parametrize("tz", ["Asia/Tokyo", "America/New_York"])
def test_find_price_steps_returns_exchange_date_for_tz_aware_index(tz: str, hour: str) -> None:
    # 取引所の日付で返す（UTC に直して1日ずれない）
    index = pd.DatetimeIndex([pd.Timestamp(f"2023-03-0{d} {hour}", tz=tz) for d in (1, 2, 3)])
    df = pd.DataFrame({"Close": [100.0, 100.0, 50.0]}, index=index)
    assert find_price_steps(df) == [("2023-03-03", 0.5)]


# --- is_pending_split -------------------------------------------------------


@pytest.mark.parametrize("ratio", [0.5, 0.1, 2.0, 10.0, 0.524, 2.09])
def test_is_pending_split_simple_ratios_are_true(ratio: float) -> None:
    assert is_pending_split(_day("2023-03-06"), ratio, _day("2023-03-07")) is True


@pytest.mark.parametrize("ratio", [1.0 / 11.0, 11.0, 0.43, 0.526, 2.11])
def test_is_pending_split_other_ratios_are_false(ratio: float) -> None:
    assert is_pending_split(_day("2023-03-06"), ratio, _day("2023-03-07")) is False


def test_is_pending_split_five_weekdays_true_six_false() -> None:
    # 月曜から翌月曜は5営業日、翌火曜は6営業日
    assert is_pending_split(_day("2023-03-06"), 0.5, _day("2023-03-13")) is True
    assert is_pending_split(_day("2023-03-06"), 0.5, _day("2023-03-14")) is False


def test_is_pending_split_weekends_are_not_counted() -> None:
    # 金曜から翌金曜は暦で7日だが5営業日
    assert is_pending_split(_day("2023-03-03"), 0.5, _day("2023-03-10")) is True
    # 金曜から翌々月曜は暦で10日、6営業日
    assert is_pending_split(_day("2023-03-03"), 0.5, _day("2023-03-13")) is False


def test_is_pending_split_step_after_latest_uses_same_window() -> None:
    assert is_pending_split(_day("2023-03-13"), 0.5, _day("2023-03-06")) is True
    assert is_pending_split(_day("2023-03-14"), 0.5, _day("2023-03-06")) is False


# --- pending_split_steps ----------------------------------------------------


def test_pending_split_steps_returns_only_pending_with_tz_aware_latest() -> None:
    # 2/27 は 0.5（最新 3/7 から6営業日で外れる）、2/28 は 2.0（5営業日で残る）、
    # 3/6 は 0.43（近いが分割の比でない）
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 50.0, 100.0, 100.0, 100.0, 100.0, 43.0, 43.0]
    df = _frame(closes, start="2023-02-20", tz="Asia/Tokyo")
    assert df.index[5].strftime("%Y-%m-%d") == "2023-02-27"
    latest = pd.Timestamp("2023-03-07 00:00", tz="Asia/Tokyo")
    assert pending_split_steps(df, latest) == [("2023-02-28", 2.0)]


def test_pending_split_steps_without_steps_is_empty() -> None:
    latest = pd.Timestamp("2023-03-03", tz="America/New_York")
    assert pending_split_steps(_frame([100.0, 101.0, 99.0]), latest) == []
