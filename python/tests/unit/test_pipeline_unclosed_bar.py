"""引けていない当日の足を market_data_raw に残さないことのテスト（#890）"""

from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.market_data.pipeline import fetch_stock_data_with_features


def _bars(start: str, end: str) -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    n = len(dates)
    return pd.DataFrame(
        {
            "Open": np.linspace(100, 120, n),
            "High": np.linspace(101, 121, n),
            "Low": np.linspace(99, 119, n),
            "Close": np.linspace(100, 120, n),
            "Volume": np.full(n, 3000),
        },
        index=dates,
    )


def _run(market: str, now_utc: datetime, db_has_rows: bool = True):
    """DB は 2026-10-21 まで、取得は 2026-10-22（当日）の足を返す形で日次の取得を動かす"""
    db_df = _bars("2026-07-01", "2026-10-21") if db_has_rows else None
    fresh = _bars("2026-10-22", "2026-10-22") if db_has_rows else _bars("2026-07-01", "2026-10-22")
    with (
        patch("src.market_data.bar_close._utcnow", return_value=now_utc),
        patch("src.market_data.pipeline.get_raw_ohlcv_from_db", return_value=db_df),
        patch("src.market_data.pipeline.should_fetch_fresh_data", return_value=True),
        patch("src.market_data.pipeline.get_stock_data", return_value=fresh),
        patch("src.market_data.pipeline.save_raw_ohlcv") as mock_save,
        patch("src.market_data.pipeline.fetch_cross_asset_features", return_value=None),
        patch("src.market_data.pipeline.fetch_additional_macro_features", return_value=None),
        patch("src.market_data.pipeline.fetch_news_sentiment_with_llm", return_value=None),
        patch("src.market_data.pipeline.fetch_pageview_features", return_value=None),
    ):
        result = fetch_stock_data_with_features(
            market=market, symbol="7203", start_date="2021-10-22", end_date="2026-10-22"
        )
    saved_dates = [d for call in mock_save.call_args_list for d in call.args[2].index]
    return result, saved_dates


_TODAY = pd.Timestamp("2026-10-22")


@pytest.mark.parametrize(
    "market, now_utc",
    [
        # 東証の場中 2026-10-22 10:00 JST
        ("jp", datetime(2026, 10, 22, 1, 0, tzinfo=timezone.utc)),
        # 米国の場中 2026-10-22 11:00 EDT（JST 10-23 00:00）
        ("us", datetime(2026, 10, 22, 15, 0, tzinfo=timezone.utc)),
    ],
)
def test_unclosed_today_bar_is_not_saved_in_diff_fetch(market, now_utc):
    result, saved_dates = _run(market, now_utc)

    assert _TODAY not in saved_dates
    assert result is not None
    assert result[2].index.max() == pd.Timestamp("2026-10-21")


@pytest.mark.parametrize(
    "market, now_utc",
    [
        # 東証の引けの後 2026-10-22 16:00 JST
        ("jp", datetime(2026, 10, 22, 7, 0, tzinfo=timezone.utc)),
        # 米国の引けの後 2026-10-22 16:30 EDT
        ("us", datetime(2026, 10, 22, 20, 30, tzinfo=timezone.utc)),
    ],
)
def test_closed_today_bar_is_saved(market, now_utc):
    result, saved_dates = _run(market, now_utc)

    assert _TODAY in saved_dates
    assert result is not None


def test_unclosed_today_bar_is_not_saved_in_full_fetch():
    result, saved_dates = _run("jp", datetime(2026, 10, 22, 1, 0, tzinfo=timezone.utc), False)

    assert saved_dates
    assert _TODAY not in saved_dates
    assert result is not None
    assert result[2].index.max() == pd.Timestamp("2026-10-21")
