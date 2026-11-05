"""米国の取引所の休場日の判定（#1608）"""

from datetime import date

import pytest

from src.domain.us_market_holidays import is_us_business_day, is_us_market_holiday

# NYSE の公表の休場日
_HOLIDAYS_2026 = [
    date(2026, 1, 1),
    date(2026, 1, 19),
    date(2026, 2, 16),
    date(2026, 4, 3),
    date(2026, 5, 25),
    date(2026, 6, 19),
    date(2026, 7, 3),
    date(2026, 9, 7),
    date(2026, 11, 26),
    date(2026, 12, 25),
]
_HOLIDAYS_2027 = [
    date(2027, 1, 1),
    date(2027, 1, 18),
    date(2027, 2, 15),
    date(2027, 3, 26),
    date(2027, 5, 31),
    date(2027, 6, 18),
    date(2027, 7, 5),
    date(2027, 9, 6),
    date(2027, 11, 25),
    date(2027, 12, 24),
]


@pytest.mark.parametrize("day", _HOLIDAYS_2026 + _HOLIDAYS_2027)
def test_listed_holidays_are_not_business_days(day):
    assert is_us_market_holiday(day) is True
    assert is_us_business_day(day) is False


@pytest.mark.parametrize("good_friday", [date(2024, 3, 29), date(2025, 4, 18), date(2028, 4, 14)])
def test_good_friday_follows_easter(good_friday):
    assert is_us_market_holiday(good_friday) is True


@pytest.mark.parametrize(
    "day",
    [
        date(2026, 7, 2),  # 振替の前の日
        date(2026, 11, 27),  # 感謝祭の次の日(半日の立ち合いがある)
        date(2026, 12, 24),
        date(2027, 12, 31),  # 2028 年の元日は土曜で、前の金曜は休まない
    ],
)
def test_ordinary_weekdays_are_business_days(day):
    assert is_us_business_day(day) is True


@pytest.mark.parametrize("day", [date(2026, 11, 7), date(2026, 11, 8)])
def test_weekends_are_not_business_days(day):
    assert is_us_business_day(day) is False
