"""クロスアセット・追加マクロの列が欠けた銘柄は特徴量を作らないことのテスト（#1491）"""

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


def _history_except(missing_ticker):
    """missing_ticker だけ空を返し、ほかは終値の足を返す ticker_history の代わり"""

    def _history(ticker, start, end):
        if ticker == missing_ticker:
            return pd.DataFrame()
        dates = pd.bdate_range("2026-05-01", "2026-10-21")
        return pd.DataFrame({"Close": np.linspace(10, 20, len(dates))}, index=dates)

    return _history


def _run(missing_ticker, caplog):
    fresh = _bars("2026-06-01", "2026-10-21")
    with (
        patch(
            "src.market_data.yf_client.ticker_history", side_effect=_history_except(missing_ticker)
        ),
        patch("src.market_data.pipeline.get_raw_ohlcv_from_db", return_value=None),
        patch("src.market_data.pipeline.should_fetch_fresh_data", return_value=True),
        patch("src.market_data.pipeline.get_stock_data", return_value=fresh),
        patch("src.market_data.pipeline.drop_unclosed_bars", side_effect=lambda df, _m: df),
        patch("src.market_data.pipeline.fetch_news_sentiment_with_llm", return_value=None),
        patch("src.market_data.pipeline.fetch_pageview_features", return_value=None),
        caplog.at_level("WARNING"),
    ):
        result = fetch_stock_data_with_features(
            market="jp",
            symbol="7203",
            start_date="2026-06-01",
            end_date="2026-10-21",
            defer_raw_save=True,
        )
    return result, fresh


@pytest.mark.parametrize(
    "missing_ticker, missing_col",
    [("^VIX", "vix_close"), ("^GSPC", "sp500_close")],
)
def test_missing_macro_ticker_skips_features_and_keeps_bars(missing_ticker, missing_col, caplog):
    result, fresh = _run(missing_ticker, caplog)

    assert result is not None
    _market, _symbol, data, raw, quality = result
    assert data is None
    assert quality is None
    # 取れた足は今までどおり保存のために返る
    pd.testing.assert_frame_equal(raw, fresh)
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("jp/7203" in m and missing_col in m for m in warnings)


def test_all_macro_tickers_fetched_builds_features(caplog):
    result, _fresh = _run(None, caplog)

    assert result is not None
    data = result[2]
    assert data is not None
    for col in ("vix_close_lag1", "usdjpy_close_lag1", "tnx_close_lag1", "sp500_close_lag1"):
        assert col in data.columns
