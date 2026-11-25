"""営業日の遅れの数え方を domain にまとめる前の値が、3つの段で変わらないことを確かめる（#2163）

期待値は、まとめる前の3つの関数を流して取った値である。1行が最後の足の日、
1文字が基準の日（窓の初日から順）で、遅れの営業日の数を 36 進の1桁で書く。
"""

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from src.market_data.pipeline import _count_weekdays_behind as market_data_count
from src.prediction.predict_unified import count_weekdays_behind as prediction_count
from src.rule_engine.pipeline import _count_weekdays_behind as rule_engine_count

_WINDOWS = {
    "spring": (date(2026, 4, 20), date(2026, 5, 15)),
    "new_year": (date(2026, 12, 25), date(2027, 1, 10)),
}

_EXPECTED = {
    ("spring", "jp"): """
        012344456678888889aaabcdef
        001233345567777778999abcde
        0001222344566666678889abcd
        00001112334555555677789abc
        000000012234444445666789ab
        000000012234444445666789ab
        000000012234444445666789ab
        0000000011233333345556789a
        00000000001222222344456789
        00000000001222222344456789
        00000000000111111233345678
        00000000000000000122234567
        00000000000000000122234567
        00000000000000000122234567
        00000000000000000122234567
        00000000000000000122234567
        00000000000000000122234567
        00000000000000000011123456
        00000000000000000000012345
        00000000000000000000012345
        00000000000000000000012345
        00000000000000000000001234
        00000000000000000000000123
        00000000000000000000000012
        00000000000000000000000001
        00000000000000000000000000
    """,
    ("new_year", "jp"): """
        00012333334567888
        00012333334567888
        00012333334567888
        00001222223456777
        00000111112345666
        00000000001234555
        00000000001234555
        00000000001234555
        00000000001234555
        00000000001234555
        00000000000123444
        00000000000012333
        00000000000001222
        00000000000000111
        00000000000000000
        00000000000000000
        00000000000000000
    """,
    ("spring", "us"): """
        01234445678999abcdeeefghij
        001233345678889abcdddefghi
        0001222345677789abcccdefgh
        00001112345666789abbbcdefg
        000000012345556789aaabcdef
        000000012345556789aaabcdef
        000000012345556789aaabcdef
        000000001234445678999abcde
        0000000001233345678889abcd
        00000000001222345677789abc
        000000000001112345666789ab
        0000000000000012345556789a
        0000000000000012345556789a
        0000000000000012345556789a
        00000000000000012344456789
        00000000000000001233345678
        00000000000000000122234567
        00000000000000000011123456
        00000000000000000000012345
        00000000000000000000012345
        00000000000000000000012345
        00000000000000000000001234
        00000000000000000000000123
        00000000000000000000000012
        00000000000000000000000001
        00000000000000000000000000
    """,
    ("new_year", "us"): """
        00012345556789aaa
        00012345556789aaa
        00012345556789aaa
        00001234445678999
        00000123334567888
        00000012223456777
        00000001112345666
        00000000001234555
        00000000001234555
        00000000001234555
        00000000000123444
        00000000000012333
        00000000000001222
        00000000000000111
        00000000000000000
        00000000000000000
        00000000000000000
    """,
}

_COUNTS = {
    "market_data": lambda last, base, market: market_data_count(
        pd.Timestamp(last), base.isoformat(), market
    ),
    "prediction": lambda last, base, market: prediction_count(
        pd.Timestamp(last), datetime(base.year, base.month, base.day), market
    ),
    "rule_engine": rule_engine_count,
}


def _digit(value: int) -> str:
    return "0123456789abcdefghijklmnopqrstuvwxyz"[value]


@pytest.mark.parametrize("window", list(_WINDOWS))
@pytest.mark.parametrize("market", ["jp", "us"])
@pytest.mark.parametrize("stage", list(_COUNTS))
def test_all_pairs_keep_previous_values(window, market, stage):
    start, end = _WINDOWS[window]
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    rows = ["".join(_digit(_COUNTS[stage](last, base, market)) for base in days) for last in days]
    expected = _EXPECTED[(window, market)].split()
    assert rows == expected, "\n" + "\n".join(rows)


def test_prediction_without_market_keeps_previous_value():
    assert prediction_count(pd.Timestamp("2026-04-28"), datetime(2026, 5, 7)) == 7


def test_prediction_with_tz_keeps_previous_value():
    last = pd.Timestamp("2026-04-28", tz="Asia/Tokyo")
    assert prediction_count(last, datetime(2026, 5, 7), "jp") == 3


def test_prediction_with_time_of_day_keeps_previous_value():
    last = pd.Timestamp("2026-04-28 15:30")
    assert prediction_count(last, datetime(2026, 5, 7, 9, 15), "jp") == 3


def test_market_data_with_tz_and_time_keeps_previous_value():
    last = pd.Timestamp("2026-04-28 23:30", tz="America/New_York")
    assert market_data_count(last, "2026-05-07", "us") == 7
