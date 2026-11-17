"""drop_unclosed_bars を直に呼ぶテスト（#890, #1962）"""

from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from src.market_data.bar_close import drop_unclosed_bars


def _bars(start: str, end: str, tz: str | None = None) -> pd.DataFrame:
    dates = pd.bdate_range(start, end, tz=tz)
    return pd.DataFrame({"Close": range(len(dates))}, index=dates)


def _drop(df, market: str, now_utc: datetime):
    with patch("src.market_data.bar_close._utcnow", return_value=now_utc):
        return drop_unclosed_bars(df, market)


_JP_OPEN = datetime(2026, 10, 22, 1, 0, tzinfo=timezone.utc)  # 10:00 JST


@pytest.mark.parametrize("df", [None, pd.DataFrame()])
def test_none_and_empty_are_returned_as_is(df):
    assert _drop(df, "jp", _JP_OPEN) is df


def test_unknown_market_is_returned_as_is():
    df = _bars("2026-10-20", "2026-10-22")

    result = _drop(df, "hk", _JP_OPEN)

    assert result is df
    assert len(result) == 3


@pytest.mark.parametrize(
    "market, now_utc, today_kept",
    [
        # 東証の引けちょうど 2026-10-22 15:30 JST は落とさない
        ("jp", datetime(2026, 10, 22, 6, 30, tzinfo=timezone.utc), True),
        # その1分前 15:29 JST は落とす
        ("jp", datetime(2026, 10, 22, 6, 29, tzinfo=timezone.utc), False),
        # 米国の引けちょうど 2026-10-22 16:00 EDT（夏時間 UTC-4）は落とさない
        ("us", datetime(2026, 10, 22, 20, 0, tzinfo=timezone.utc), True),
        # その1分前 15:59 EDT は落とす
        ("us", datetime(2026, 10, 22, 19, 59, tzinfo=timezone.utc), False),
    ],
)
def test_close_time_boundary(market, now_utc, today_kept):
    df = _bars("2026-10-20", "2026-10-22")

    result = _drop(df, market, now_utc)

    assert (pd.Timestamp("2026-10-22") in result.index) is today_kept
    assert pd.Timestamp("2026-10-21") in result.index


@pytest.mark.parametrize(
    "now_utc, today_kept",
    [
        # 冬時間（EST, UTC-5）の 2026-11-17 16:00 EST = 21:00 UTC は落とさない
        (datetime(2026, 11, 17, 21, 0, tzinfo=timezone.utc), True),
        # 15:59 EST は落とす
        (datetime(2026, 11, 17, 20, 59, tzinfo=timezone.utc), False),
        # 20:30 UTC は夏時間なら引けの後だが、冬時間では 15:30 EST の場中なので落とす
        (datetime(2026, 11, 17, 20, 30, tzinfo=timezone.utc), False),
    ],
)
def test_us_close_line_follows_winter_time(now_utc, today_kept):
    df = _bars("2026-11-13", "2026-11-17")

    result = _drop(df, "us", now_utc)

    assert (pd.Timestamp("2026-11-17") in result.index) is today_kept


def test_us_summer_time_after_close_in_utc_terms_is_kept():
    # 20:30 UTC は夏時間（EDT）では 16:30 の引けの後なので落とさない
    df = _bars("2026-10-20", "2026-10-22")

    result = _drop(df, "us", datetime(2026, 10, 22, 20, 30, tzinfo=timezone.utc))

    assert pd.Timestamp("2026-10-22") in result.index


@pytest.mark.parametrize(
    "market, tz, now_utc",
    [
        # 米国の場中 2026-10-22 11:00 EDT、index は America/New_York
        ("us", "America/New_York", datetime(2026, 10, 22, 15, 0, tzinfo=timezone.utc)),
        # 東証の場中 2026-10-22 10:00 JST、index は Asia/Tokyo
        ("jp", "Asia/Tokyo", _JP_OPEN),
    ],
)
def test_tz_aware_index_drops_only_today_and_after(market, tz, now_utc):
    df = _bars("2026-10-19", "2026-10-23", tz=tz)

    result = _drop(df, market, now_utc)

    days = [d.strftime("%Y-%m-%d") for d in result.index]
    assert days == ["2026-10-19", "2026-10-20", "2026-10-21"]
    assert result.index.tz is not None
