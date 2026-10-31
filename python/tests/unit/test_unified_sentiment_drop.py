"""取得元のキーが無い市場では、センチメントの列を統合の特徴量に入れない（#1417）。"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.prediction.unified_model_pipeline import prepare_unified_features

_SENTIMENT_COLS = [
    f"{base}_lag{i}"
    for base in ("sentiment_score", "sentiment_ma5", "sentiment_momentum", "news_count")
    for i in range(1, 11)
]


def _frame(market: str) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "market": [market] * 3,
            "symbol": ["A"] * 3,
            "date": pd.to_datetime(["2026-10-27", "2026-10-28", "2026-10-29"]),
            "Close": [1.0, 2.0, 3.0],
            "y": [0.1, 0.2, 0.3],
        }
    )
    for col in _SENTIMENT_COLS:
        df[col] = 0.0
    return df


@pytest.fixture(autouse=True)
def _no_exclusions():
    with patch("src.prediction.unified_model_pipeline.load_excluded_features", return_value=[]):
        yield


def test_us_without_newsapi_key_drops_sentiment(monkeypatch):
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    monkeypatch.delenv("EDINET_API_KEY", raising=False)
    X, _ = prepare_unified_features(_frame("us"))
    assert [c for c in X.columns if c in _SENTIMENT_COLS] == []
    assert "Close" in X.columns


def test_jp_with_edinet_key_keeps_sentiment(monkeypatch):
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    monkeypatch.setenv("EDINET_API_KEY", "dummy")
    X, _ = prepare_unified_features(_frame("jp"))
    assert all(c in X.columns for c in _SENTIMENT_COLS)


def test_jp_without_edinet_key_drops_sentiment(monkeypatch):
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    monkeypatch.delenv("EDINET_API_KEY", raising=False)
    X, _ = prepare_unified_features(_frame("jp"))
    assert [c for c in X.columns if c in _SENTIMENT_COLS] == []


def test_mixed_markets_keep_sentiment_when_jp_has_key(monkeypatch):
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    monkeypatch.setenv("EDINET_API_KEY", "dummy")
    df = pd.concat([_frame("us"), _frame("jp")], ignore_index=True)
    X, _ = prepare_unified_features(df)
    assert all(c in X.columns for c in _SENTIMENT_COLS)
