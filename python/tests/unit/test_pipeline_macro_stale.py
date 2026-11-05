"""相場の指標の列が止まった銘柄は特徴量を作らないことのテスト（#1622）"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.market_data.macro_join import stale_macro_columns
from src.market_data.pipeline import fetch_stock_data_with_features

_LAST_BAR = "2026-10-21"  # 水曜
_INSIDE_LINE = "2026-10-16"  # 金曜。平日で3日古い（線ちょうど）
_OUTSIDE_LINE = "2026-10-15"  # 木曜。平日で4日古い


def _bars() -> pd.DataFrame:
    dates = pd.bdate_range("2026-06-01", _LAST_BAR)
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


def _history_stopped_at(stopped_ticker, last_date):
    """stopped_ticker だけ last_date で止まった足を返す ticker_history の代わり"""

    def _history(ticker, start, end):
        stop = last_date if ticker == stopped_ticker else _LAST_BAR
        dates = pd.bdate_range("2026-05-01", stop)
        return pd.DataFrame({"Close": np.linspace(10, 20, len(dates))}, index=dates)

    return _history


def _run(stopped_ticker, last_date, caplog):
    fresh = _bars()
    with (
        patch(
            "src.market_data.yf_client.ticker_history",
            side_effect=_history_stopped_at(stopped_ticker, last_date),
        ),
        patch("src.market_data.pipeline.get_raw_ohlcv_from_db", return_value=None),
        patch("src.market_data.pipeline.should_fetch_fresh_data", return_value=True),
        patch("src.market_data.pipeline.get_stock_data", return_value=fresh),
        patch("src.market_data.pipeline.drop_unclosed_bars", side_effect=lambda df, _m: df),
        patch("src.market_data.pipeline.fetch_news_sentiment_with_llm", return_value=None),
        patch("src.market_data.pipeline.fetch_pageview_features", return_value=None),
        patch("src.market_data.pipeline.resolve_article", return_value=None),
        caplog.at_level("WARNING"),
    ):
        result = fetch_stock_data_with_features(
            market="jp",
            symbol="7203",
            start_date="2026-06-01",
            end_date=_LAST_BAR,
            defer_raw_save=True,
        )
    return result, fresh


@pytest.mark.parametrize(
    "ticker, col",
    [
        ("^VIX", "vix_close"),
        ("JPY=X", "usdjpy_close"),
        ("^TNX", "tnx_close"),
        ("^GSPC", "sp500_close"),
        ("DX-Y.NYB", "dxy_close"),
        ("GC=F", "gold_close"),
    ],
)
def test_stopped_macro_column_skips_features_and_keeps_bars(ticker, col, caplog):
    """クロスアセットと追加マクロのどの列が線より古くても、特徴量を作らず足は返す"""
    result, fresh = _run(ticker, _OUTSIDE_LINE, caplog)

    assert result is not None
    _market, _symbol, data, raw, quality = result
    assert data is None
    assert quality is None
    pd.testing.assert_frame_equal(raw, fresh)
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("jp/7203" in m and col in m and _OUTSIDE_LINE in m for m in warnings)


@pytest.mark.parametrize("ticker", ["^VIX", "GC=F"])
@pytest.mark.parametrize("last_date", ["2026-10-20", "2026-10-19", _INSIDE_LINE])
def test_gap_inside_line_builds_features(ticker, last_date, caplog):
    """休場の1〜2日と、線ちょうどの遅れでは、今までどおり特徴量ができる"""
    result, _fresh = _run(ticker, last_date, caplog)

    assert result is not None
    assert result[2] is not None
    assert "vix_close_lag1" in result[2].columns


def _frame(columns: dict[str, str]) -> pd.DataFrame:
    """列ごとの最後の日までを値で埋め、その後を NaN にした表（外側結合の形）"""
    dates = pd.bdate_range("2026-10-01", _LAST_BAR)
    return pd.DataFrame(
        {
            c: [1.0 if d <= pd.Timestamp(last) else np.nan for d in dates]
            for c, last in columns.items()
        },
        index=dates,
    )


def _whole_frames(**overrides: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cross = {c: _LAST_BAR for c in ("vix_close", "usdjpy_close", "tnx_close")}
    additional = {c: _LAST_BAR for c in ("sp500_close", "dxy_close", "gold_close")}
    return (
        _frame({c: overrides.get(c, last) for c, last in cross.items()}),
        _frame({c: overrides.get(c, last) for c, last in additional.items()}),
    )


def test_stale_columns_are_found_in_both_groups():
    cross, additional = _whole_frames(tnx_close=_OUTSIDE_LINE, dxy_close="2026-10-02")

    assert stale_macro_columns(cross, additional, pd.Timestamp(_LAST_BAR)) == {
        "tnx_close": _OUTSIDE_LINE,
        "dxy_close": "2026-10-02",
    }


def test_column_with_no_value_counts_as_stale():
    cross, additional = _whole_frames()
    cross["vix_close"] = np.nan

    assert stale_macro_columns(cross, additional, pd.Timestamp(_LAST_BAR)) == {"vix_close": "なし"}


def test_timezone_aware_last_bar_is_compared_by_date():
    cross, additional = _whole_frames()

    last_bar = pd.Timestamp(_LAST_BAR, tz="Asia/Tokyo")
    assert stale_macro_columns(cross, additional, last_bar) == {}


def test_stopped_bars_of_symbol_are_not_blamed_on_macro():
    """銘柄の足の方が古いときは、相場の指標は古いと数えない"""
    cross, additional = _whole_frames()

    assert stale_macro_columns(cross, additional, pd.Timestamp("2026-10-01")) == {}
