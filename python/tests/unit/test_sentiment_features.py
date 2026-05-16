import unittest

import numpy as np
import pandas as pd

from src.features.sentiment_features import _score_titles, add_sentiment_features


def _make_ohlcv(n: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open": np.linspace(100, 130, n).astype(float),
            "High": np.linspace(101, 131, n).astype(float),
            "Low": np.linspace(99, 129, n).astype(float),
            "Close": np.linspace(100, 130, n).astype(float),
            "Volume": np.random.randint(1000, 2000, n).astype(int),
        },
        index=dates,
    )


def _make_sentiment_df(n: int = 30, score: float = 0.5) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"sentiment_score": [score] * n, "news_count": [10] * n},
        index=dates,
    )


class TestAddSentimentFeaturesPlaceholder(unittest.TestCase):
    """sentiment_df を渡さない場合は中立プレースホルダーが設定される"""

    def setUp(self):
        self.df = _make_ohlcv(30)

    def test_columns_added(self):
        result = add_sentiment_features(self.df)
        self.assertIn("sentiment_score", result.columns)
        self.assertIn("sentiment_ma5", result.columns)
        self.assertIn("sentiment_momentum", result.columns)
        self.assertIn("news_count", result.columns)

    def test_placeholder_neutral(self):
        result = add_sentiment_features(self.df)
        self.assertTrue((result["sentiment_score"] == 0.0).all())
        self.assertTrue((result["news_count"] == 0).all())

    def test_momentum_zero_for_constant_score(self):
        result = add_sentiment_features(self.df)
        self.assertTrue((result["sentiment_momentum"] == 0.0).all())

    def test_input_not_mutated(self):
        orig_cols = set(self.df.columns)
        add_sentiment_features(self.df)
        self.assertEqual(orig_cols, set(self.df.columns))


class TestAddSentimentFeaturesWithData(unittest.TestCase):
    """外部 sentiment_df を渡した場合の動作"""

    def setUp(self):
        self.df = _make_ohlcv(30)

    def test_sentiment_score_merged(self):
        sent = _make_sentiment_df(30, score=0.8)
        result = add_sentiment_features(self.df, sentiment_df=sent)
        self.assertTrue((result["sentiment_score"] == 0.8).all())

    def test_news_count_merged(self):
        sent = _make_sentiment_df(30, score=0.5)
        result = add_sentiment_features(self.df, sentiment_df=sent)
        self.assertTrue((result["news_count"] == 10).all())

    def test_ma_window(self):
        sent = _make_sentiment_df(30, score=1.0)
        result = add_sentiment_features(self.df, sentiment_df=sent, window_days=3)
        self.assertIn("sentiment_ma3", result.columns)
        self.assertNotIn("sentiment_ma5", result.columns)
        self.assertAlmostEqual(result["sentiment_ma3"].iloc[-1], 1.0)

    def test_missing_dates_filled_neutral(self):
        # sentiment_df が 10 行だけの場合、残りは 0.0 で補完される
        sent = _make_sentiment_df(10, score=0.6)
        result = add_sentiment_features(self.df, sentiment_df=sent)
        self.assertEqual(result["sentiment_score"].isna().sum(), 0)
        self.assertAlmostEqual(result["sentiment_score"].iloc[0], 0.6)
        self.assertAlmostEqual(result["sentiment_score"].iloc[-1], 0.0)


class TestScoreTitles(unittest.TestCase):
    def test_positive_titles(self):
        titles = ["Stock surge to record highs", "Company beats expectations rally"]
        score = _score_titles(titles)
        self.assertGreater(score, 0.0)

    def test_negative_titles(self):
        titles = ["Market crash and decline continues", "Stock drop miss earnings"]
        score = _score_titles(titles)
        self.assertLess(score, 0.0)

    def test_empty_titles(self):
        self.assertEqual(_score_titles([]), 0.0)

    def test_neutral_titles(self):
        titles = ["Company announces quarterly results", "CEO speaks at conference"]
        score = _score_titles(titles)
        self.assertEqual(score, 0.0)

    def test_score_range(self):
        titles = ["surge jump beat record rally gain rise bullish strong"] * 5
        score = _score_titles(titles)
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
