import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.market_data.sentiment_features import (
    _fetch_news_articles,
    _records_to_sentiment_df,
    _score_titles,
    add_sentiment_features,
    fetch_news_sentiment,
    fetch_news_sentiment_with_llm,
)


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


class TestFetchNewsArticles(unittest.TestCase):
    """_fetch_news_articles の単体テスト"""

    def test_returns_none_when_no_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            result = _fetch_news_articles("AAPL", "2024-01-01", "2024-01-31")
        self.assertIsNone(result)

    @patch("src.market_data.sentiment_features._fetch_news_articles")
    def test_fetch_news_sentiment_returns_none_when_no_articles(self, mock_fetch):
        mock_fetch.return_value = None
        result = fetch_news_sentiment("AAPL", "2024-01-01", "2024-01-31")
        self.assertIsNone(result)

    @patch("src.market_data.sentiment_features._fetch_news_articles")
    def test_fetch_news_sentiment_returns_df_when_articles_exist(self, mock_fetch):
        mock_fetch.return_value = {"2024-01-02": ["Apple stock surges on record revenue"]}
        result = fetch_news_sentiment("AAPL", "2024-01-01", "2024-01-31")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("sentiment_score", result.columns)

    @patch("requests.get")
    def test_fetch_news_articles_returns_records_when_api_key_set(self, mock_get):
        """NEWSAPI_KEY が設定されている場合に日付別記事辞書を返すこと"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "articles": [
                {"publishedAt": "2024-01-02T10:00:00Z", "title": "Stock surges"},
                {"publishedAt": "2024-01-02T12:00:00Z", "title": "Record earnings"},
            ]
        }
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {"NEWSAPI_KEY": "test_key"}):
            result = _fetch_news_articles("AAPL", "2024-01-01", "2024-01-31")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("2024-01-02", result)
        self.assertEqual(len(result["2024-01-02"]), 2)

    @patch("requests.get")
    def test_fetch_news_articles_returns_none_when_empty_articles(self, mock_get):
        """API は成功するが articles が空の場合は None を返すこと"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"articles": []}
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {"NEWSAPI_KEY": "test_key"}):
            result = _fetch_news_articles("AAPL", "2024-01-01", "2024-01-31")

        self.assertIsNone(result)

    @patch("requests.get")
    def test_fetch_news_articles_returns_none_on_request_error(self, mock_get):
        """HTTP リクエストが失敗した場合は None を返すこと"""
        mock_get.side_effect = ConnectionError("network error")

        with patch.dict("os.environ", {"NEWSAPI_KEY": "test_key"}):
            result = _fetch_news_articles("AAPL", "2024-01-01", "2024-01-31")

        self.assertIsNone(result)

    def test_records_to_sentiment_df_structure(self):
        records = {
            "2024-01-02": ["Stock surges", "Bullish outlook"],
            "2024-01-03": ["Market declines"],
        }
        df = _records_to_sentiment_df(records, _score_titles)
        self.assertIsNotNone(df)
        assert df is not None
        self.assertIn("sentiment_score", df.columns)
        self.assertIn("news_count", df.columns)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.loc[pd.Timestamp("2024-01-02"), "news_count"], 2)

    def test_records_to_sentiment_df_empty_returns_none(self):
        result = _records_to_sentiment_df({}, _score_titles)
        self.assertIsNone(result)


class TestFetchNewsSentimentWithLlm(unittest.TestCase):
    """fetch_news_sentiment_with_llm の LLM パス / フォールバックテスト（US 市場）"""

    @patch("src.market_data.sentiment_features._fetch_news_articles")
    def test_returns_none_when_no_articles(self, mock_fetch):
        mock_fetch.return_value = None
        result = fetch_news_sentiment_with_llm("AAPL", "2024-01-01", "2024-01-31")
        self.assertIsNone(result)

    @patch("src.market_data.sentiment_features._fetch_news_articles")
    @patch("src.market_data.adapters.llm_sentiment.OllamaClient")
    def test_uses_llm_when_available(self, mock_client_cls, mock_fetch):
        """Ollama が接続可能なとき LLM スコアが使われる"""
        mock_fetch.return_value = {"2024-01-02": ["Apple surges on record earnings"]}

        mock_client = MagicMock()
        mock_client.is_available = True
        mock_client.score.return_value = 0.85
        mock_client_cls.return_value = mock_client

        result = fetch_news_sentiment_with_llm("AAPL", "2024-01-01", "2024-01-31")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result["sentiment_score"].iloc[0], 0.85)
        mock_client.score.assert_called_once()

    @patch("src.market_data.sentiment_features._fetch_news_articles")
    @patch("src.market_data.adapters.llm_sentiment.OllamaClient")
    def test_falls_back_to_keywords_when_ollama_unavailable(self, mock_client_cls, mock_fetch):
        """Ollama が接続できないときキーワードマッチにフォールバックする"""
        mock_fetch.return_value = {
            "2024-01-02": ["Stock surge to record highs", "Company beats expectations"]
        }

        mock_client = MagicMock()
        mock_client.is_available = False
        mock_client_cls.return_value = mock_client

        result = fetch_news_sentiment_with_llm("AAPL", "2024-01-01", "2024-01-31")

        self.assertIsNotNone(result)
        assert result is not None
        # キーワードマッチ（正の語が含まれる）→ スコアが正
        self.assertGreater(result["sentiment_score"].iloc[0], 0.0)
        mock_client.score.assert_not_called()

    @patch("src.market_data.sentiment_features._fetch_news_articles")
    @patch("src.market_data.adapters.llm_sentiment.OllamaClient")
    def test_falls_back_to_keywords_when_llm_returns_none(self, mock_client_cls, mock_fetch):
        """LLM が None を返した場合（タイムアウト等）にキーワードマッチで補完する"""
        mock_fetch.return_value = {"2024-01-02": ["Market crash and decline continues"]}

        mock_client = MagicMock()
        mock_client.is_available = True
        mock_client.score.return_value = None  # LLM 失敗
        mock_client_cls.return_value = mock_client

        result = fetch_news_sentiment_with_llm("AAPL", "2024-01-01", "2024-01-31")

        self.assertIsNotNone(result)
        assert result is not None
        # キーワードマッチ（負の語が含まれる）→ スコアが負
        self.assertLess(result["sentiment_score"].iloc[0], 0.0)

    @patch("src.market_data.sentiment_features._fetch_news_articles")
    @patch("src.market_data.adapters.llm_sentiment.OllamaClient")
    def test_score_clamped_to_valid_range(self, mock_client_cls, mock_fetch):
        """LLM スコアが -1.0〜1.0 の範囲に収まる"""
        mock_fetch.return_value = {"2024-01-02": ["some news"]}

        mock_client = MagicMock()
        mock_client.is_available = True
        mock_client.score.return_value = 0.75
        mock_client_cls.return_value = mock_client

        result = fetch_news_sentiment_with_llm("AAPL", "2024-01-01", "2024-01-31")

        assert result is not None
        score = result["sentiment_score"].iloc[0]
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)

    @patch("src.market_data.sentiment_features._fetch_news_articles")
    @patch("src.market_data.adapters.llm_sentiment.OllamaClient")
    def test_multiple_dates_scored_independently(self, mock_client_cls, mock_fetch):
        """複数日のニュースが日付別に独立してスコアリングされる"""
        mock_fetch.return_value = {
            "2024-01-02": ["Bullish news"],
            "2024-01-03": ["Bearish news"],
            "2024-01-04": ["Neutral news"],
        }

        mock_client = MagicMock()
        mock_client.is_available = True
        mock_client.score.side_effect = [0.8, -0.6, 0.0]
        mock_client_cls.return_value = mock_client

        result = fetch_news_sentiment_with_llm("AAPL", "2024-01-01", "2024-01-31")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(len(result), 3)
        self.assertEqual(mock_client.score.call_count, 3)


class TestFetchNewsSentimentWithLlmJp(unittest.TestCase):
    """fetch_news_sentiment_with_llm の JP 市場（EDINET）パステスト"""

    @patch("src.market_data.sentiment_features._fetch_jp_disclosure_titles")
    def test_returns_none_when_no_edinet_data(self, mock_fetch_jp):
        mock_fetch_jp.return_value = None
        result = fetch_news_sentiment_with_llm("7203", "2024-01-01", "2024-01-31", market="jp")
        self.assertIsNone(result)
        mock_fetch_jp.assert_called_once_with("7203", "2024-01-01", "2024-01-31")

    @patch("src.market_data.sentiment_features._fetch_jp_disclosure_titles")
    @patch("src.market_data.adapters.llm_sentiment.OllamaClient")
    def test_jp_market_uses_edinet_titles(self, mock_client_cls, mock_fetch_jp):
        """JP 市場では EDINET 開示タイトルが LLM スコアリングに使われる"""
        mock_fetch_jp.return_value = {"2024-01-02": ["上方修正のお知らせ"]}

        mock_client = MagicMock()
        mock_client.is_available = True
        mock_client.score.return_value = 0.7
        mock_client_cls.return_value = mock_client

        result = fetch_news_sentiment_with_llm("7203", "2024-01-01", "2024-01-31", market="jp")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result["sentiment_score"].iloc[0], 0.7)
        mock_fetch_jp.assert_called_once_with("7203", "2024-01-01", "2024-01-31")

    @patch("src.market_data.sentiment_features._fetch_news_articles")
    @patch("src.market_data.sentiment_features._fetch_jp_disclosure_titles")
    def test_us_market_does_not_call_edinet(self, mock_fetch_jp, mock_fetch_us):
        """US 市場では EDINET を呼び出さない"""
        mock_fetch_us.return_value = None
        fetch_news_sentiment_with_llm("AAPL", "2024-01-01", "2024-01-31", market="us")
        mock_fetch_jp.assert_not_called()
        mock_fetch_us.assert_called_once()

    @patch("src.market_data.sentiment_features._fetch_jp_disclosure_titles")
    @patch("src.market_data.adapters.llm_sentiment.OllamaClient")
    def test_jp_market_falls_back_to_keywords_when_ollama_unavailable(
        self, mock_client_cls, mock_fetch_jp
    ):
        """Ollama 未接続時はキーワードマッチにフォールバックする（JP 市場）"""
        mock_fetch_jp.return_value = {"2024-01-02": ["record surge bullish"]}

        mock_client = MagicMock()
        mock_client.is_available = False
        mock_client_cls.return_value = mock_client

        result = fetch_news_sentiment_with_llm("7203", "2024-01-01", "2024-01-31", market="jp")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertGreater(result["sentiment_score"].iloc[0], 0.0)


class TestBuildScoreFnNoLlm(unittest.TestCase):
    """_build_score_fn(use_llm=False) のテスト"""

    def test_returns_keyword_fn_when_use_llm_false(self):
        from src.market_data.sentiment_features import _build_score_fn, _score_titles

        fn = _build_score_fn(use_llm=False)
        self.assertIs(fn, _score_titles)

    def test_keyword_fn_scores_positive(self):
        from src.market_data.sentiment_features import _build_score_fn

        fn = _build_score_fn(use_llm=False)
        score = fn(["stocks surge record high rally"])
        self.assertGreater(score, 0.0)


class TestFetchJpDisclosureTitlesException(unittest.TestCase):
    """_fetch_jp_disclosure_titles の例外パスのテスト"""

    @patch("src.market_data.adapters.edinet_client.EdinetClient")
    def test_returns_none_on_edinet_exception(self, mock_cls):
        from src.market_data.sentiment_features import _fetch_jp_disclosure_titles

        mock_cls.side_effect = RuntimeError("connection error")
        result = _fetch_jp_disclosure_titles("7203", "2025-01-01", "2025-01-31")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
