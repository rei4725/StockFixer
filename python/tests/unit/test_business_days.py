"""domain.business_days の営業日の遅れの数え方のテスト（#2163）"""

from datetime import date

import pytest

from src.domain.business_days import count_business_days_behind


@pytest.mark.parametrize(
    "last_day, base_day, market, expected",
    [
        # 2027 年の連休（4/29・5/3〜5/5 が休場）
        (date(2027, 4, 30), date(2027, 5, 6), "jp", 1),
        (date(2027, 4, 30), date(2027, 5, 5), "jp", 0),
        (date(2027, 4, 28), date(2027, 5, 6), "jp", 2),
        (date(2027, 4, 30), date(2027, 5, 6), "us", 4),
        # 土日は数えない（2026-10-23 は金曜）
        (date(2026, 10, 23), date(2026, 10, 25), "jp", 0),
        (date(2026, 10, 23), date(2026, 10, 25), "us", 0),
        (date(2026, 10, 23), date(2026, 10, 26), "us", 1),
        # us の祝日（2026-11-26 感謝祭）は除かない
        (date(2026, 11, 25), date(2026, 11, 27), "us", 2),
        # 市場が None なら土日だけを除く
        (date(2027, 4, 30), date(2027, 5, 6), None, 4),
    ],
)
def test_counts_business_days(last_day, base_day, market, expected):
    assert count_business_days_behind(last_day, base_day, market) == expected


def test_market_defaults_to_weekdays_only():
    assert count_business_days_behind(date(2027, 4, 30), date(2027, 5, 6)) == 4


@pytest.mark.parametrize("market", ["jp", "us", None])
@pytest.mark.parametrize("last_day", [date(2026, 10, 22), date(2026, 10, 23), date(2026, 12, 1)])
def test_last_day_on_or_after_base_day_is_zero(last_day, market):
    assert count_business_days_behind(last_day, date(2026, 10, 22), market) == 0
