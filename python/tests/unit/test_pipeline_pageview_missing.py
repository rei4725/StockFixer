"""記事のある銘柄でページビューが取れないか止まった日は特徴量を作らないテスト（#1500・#1636）"""

from unittest.mock import patch

import numpy as np
import pandas as pd

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


def _history(ticker, start, end):
    dates = pd.bdate_range("2026-05-01", "2026-10-21")
    return pd.DataFrame({"Close": np.linspace(10, 20, len(dates))}, index=dates)


def _pageviews(last_day: str) -> pd.DataFrame:
    dates = pd.date_range("2026-06-01", last_day, name="date")
    return pd.DataFrame({"pageviews": np.full(len(dates), 1000)}, index=dates)


def _run(market, symbol, caplog, pageview_df=None, end_date="2026-10-21"):
    fresh = _bars("2026-06-01", end_date)
    with (
        patch("src.market_data.yf_client.ticker_history", side_effect=_history),
        patch("src.market_data.pipeline.get_raw_ohlcv_from_db", return_value=None),
        patch("src.market_data.pipeline.should_fetch_fresh_data", return_value=True),
        patch("src.market_data.pipeline.get_stock_data", return_value=fresh),
        patch("src.market_data.pipeline.drop_unclosed_bars", side_effect=lambda df, _m: df),
        patch("src.market_data.pipeline.fetch_news_sentiment_with_llm", return_value=None),
        patch("src.market_data.pipeline.fetch_pageview_features", return_value=pageview_df),
        patch("src.market_data.pipeline.upsert_stock_features") as mock_upsert,
        caplog.at_level("WARNING"),
    ):
        result = fetch_stock_data_with_features(
            market=market,
            symbol=symbol,
            start_date="2026-06-01",
            end_date=end_date,
            defer_raw_save=True,
        )
    return result, fresh, mock_upsert


def test_article_symbol_without_pageviews_keeps_previous_rows(caplog):
    result, fresh, mock_upsert = _run("us", "AAPL", caplog)

    assert result is not None
    _market, _symbol, data, raw, quality = result
    assert data is None
    assert quality is None
    # 取れた足は今までどおり保存のために返り、特徴量の表は書き直さない
    pd.testing.assert_frame_equal(raw, fresh)
    mock_upsert.assert_not_called()
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("us/AAPL" in m and "ページビュー" in m for m in warnings)


def test_symbol_without_article_fills_zero_and_builds_features(caplog):
    result, _fresh, _mock_upsert = _run("us", "ZZZZ", caplog)

    assert result is not None
    data = result[2]
    assert data is not None
    assert (data["pageviews_lag1"] == 0.0).all()


def test_stale_pageviews_keep_previous_rows_and_warn(caplog):
    # 水曜の足に対して前の週の木曜で止まった表は、金・月・火・水の4平日遅れで線を超える
    result, fresh, mock_upsert = _run("us", "AAPL", caplog, _pageviews("2026-10-15"))

    assert result is not None
    _market, _symbol, data, raw, quality = result
    assert data is None
    assert quality is None
    pd.testing.assert_frame_equal(raw, fresh)
    mock_upsert.assert_not_called()
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("us/AAPL" in m and "2026-10-15" in m for m in warnings)


def test_pageviews_within_line_build_features_with_trailing_zero(caplog):
    # 金曜で止まった表は月・火・水の3平日遅れで線の内側。末尾は今までどおり0で埋まる
    result, _fresh, _mock_upsert = _run("us", "AAPL", caplog, _pageviews("2026-10-16"))

    assert result is not None
    data = result[2]
    assert data is not None
    assert data["pageviews_lag1"].iloc[-1] == 0.0
    assert data["pageviews_lag1"].iloc[-10] == 1000.0


def test_pageviews_past_friday_bar_build_features(caplog):
    # 金曜の足に対して土日まである表は遅れていない
    result, _fresh, _mock_upsert = _run(
        "us", "AAPL", caplog, _pageviews("2026-10-18"), end_date="2026-10-16"
    )

    assert result is not None
    assert result[2] is not None


def test_pageviews_stopped_on_sunday_count_only_weekdays(caplog):
    # 日曜で止まった表は水曜の足に対して月・火・水の3平日遅れで、週末の日を数えず止めない
    result, _fresh, _mock_upsert = _run("us", "AAPL", caplog, _pageviews("2026-10-18"))

    assert result is not None
    assert result[2] is not None
