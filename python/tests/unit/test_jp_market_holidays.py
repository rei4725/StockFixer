"""東証の休場日の表のテスト（#1050）"""

from datetime import date

from src.domain.jp_market_holidays import (
    _HOLIDAYS_BY_YEAR,
    is_jp_business_day,
    is_jp_market_holiday,
)


def test_table_holds_only_weekdays_of_its_year():
    for year, days in _HOLIDAYS_BY_YEAR.items():
        for day in days:
            assert day.year == year
            assert day.weekday() < 5, day


def test_holiday_monday_is_not_business_day():
    assert is_jp_market_holiday(date(2026, 11, 23))
    assert not is_jp_business_day(date(2026, 11, 23))


def test_substitute_holiday_is_not_business_day():
    assert not is_jp_business_day(date(2026, 5, 6))
    assert not is_jp_business_day(date(2027, 3, 22))


def test_weekend_and_plain_weekday():
    assert not is_jp_business_day(date(2026, 10, 24))
    assert is_jp_business_day(date(2026, 10, 23))


def test_year_without_table_skips_only_weekends():
    assert not is_jp_market_holiday(date(2030, 1, 1))
    assert is_jp_business_day(date(2030, 1, 1))
