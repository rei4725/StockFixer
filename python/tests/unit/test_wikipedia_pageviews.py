import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.market_data.adapters.wikipedia_pageviews import WikipediaPageviewsClient, resolve_article
from src.market_data.wikipedia_features import add_pageview_features, fetch_pageview_features


class TestResolveArticle(unittest.TestCase):
    def test_known_us_symbol(self):
        self.assertEqual(resolve_article("us", "AAPL"), ("en.wikipedia", "Apple_Inc."))

    def test_known_jp_symbol(self):
        self.assertEqual(resolve_article("jp", "7203"), ("ja.wikipedia", "トヨタ自動車"))

    def test_strips_whitespace(self):
        self.assertEqual(resolve_article("us", " AAPL "), ("en.wikipedia", "Apple_Inc."))

    def test_unknown_symbol_returns_none(self):
        self.assertIsNone(resolve_article("us", "ZZZZ"))

    def test_unknown_market_returns_none(self):
        self.assertIsNone(resolve_article("xx", "AAPL"))


class TestFetchDailyViews(unittest.TestCase):
    def setUp(self):
        self.client = WikipediaPageviewsClient()

    def _mock_response(self, items: list, status_code: int = 200) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"items": items}
        return mock_resp

    @patch("requests.get")
    def test_returns_views_by_date(self, mock_get):
        mock_get.return_value = self._mock_response(
            [
                {"timestamp": "2024010100", "views": 12345},
                {"timestamp": "2024010200", "views": 23456},
            ]
        )
        result = self.client.fetch_daily_views("us", "AAPL", "2024-01-01", "2024-01-02")
        self.assertEqual(result, {"2024-01-01": 12345, "2024-01-02": 23456})

    @patch("requests.get")
    def test_unmapped_symbol_skips_http(self, mock_get):
        result = self.client.fetch_daily_views("us", "ZZZZ", "2024-01-01", "2024-01-02")
        self.assertIsNone(result)
        mock_get.assert_not_called()

    @patch("requests.get")
    def test_404_returns_none(self, mock_get):
        mock_get.return_value = self._mock_response([], status_code=404)
        result = self.client.fetch_daily_views("us", "AAPL", "2024-01-01", "2024-01-02")
        self.assertIsNone(result)

    @patch("requests.get")
    def test_empty_items_returns_none(self, mock_get):
        mock_get.return_value = self._mock_response([])
        result = self.client.fetch_daily_views("us", "AAPL", "2024-01-01", "2024-01-02")
        self.assertIsNone(result)

    @patch("requests.get")
    def test_invalid_date_returns_none(self, mock_get):
        result = self.client.fetch_daily_views("us", "AAPL", "not-a-date", "2024-01-02")
        self.assertIsNone(result)
        mock_get.assert_not_called()

    @patch("requests.get")
    def test_user_agent_header_is_sent(self, mock_get):
        mock_get.return_value = self._mock_response([{"timestamp": "2024010100", "views": 1}])
        self.client.fetch_daily_views("us", "AAPL", "2024-01-01", "2024-01-01")
        _, kwargs = mock_get.call_args
        self.assertIn("User-Agent", kwargs.get("headers", {}))


class TestFetchPageviewFeatures(unittest.TestCase):
    @patch(
        "src.market_data.adapters.wikipedia_pageviews.WikipediaPageviewsClient.fetch_daily_views"
    )
    def test_builds_dataframe(self, mock_fetch):
        mock_fetch.return_value = {"2024-01-02": 200, "2024-01-01": 100}
        df = fetch_pageview_features("AAPL", "us", "2024-01-01", "2024-01-02")
        self.assertIsNotNone(df)
        assert df is not None
        self.assertEqual(list(df.columns), ["pageviews"])
        # ソートされて昇順になっていること
        self.assertEqual(df.index[0], pd.Timestamp("2024-01-01"))
        self.assertEqual(df["pageviews"].iloc[0], 100)

    @patch(
        "src.market_data.adapters.wikipedia_pageviews.WikipediaPageviewsClient.fetch_daily_views"
    )
    def test_none_when_no_records(self, mock_fetch):
        mock_fetch.return_value = None
        self.assertIsNone(fetch_pageview_features("AAPL", "us", "2024-01-01", "2024-01-02"))


class TestAddPageviewFeatures(unittest.TestCase):
    def _base_df(self, n: int = 10) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        return pd.DataFrame({"close": np.arange(n, dtype=float)}, index=idx)

    def test_adds_all_columns_with_external_data(self):
        df = self._base_df()
        pageview_df = pd.DataFrame({"pageviews": np.arange(100, 110, dtype=float)}, index=df.index)
        out = add_pageview_features(df, pageview_df=pageview_df)
        for col in ("pageviews", "pageviews_ma7", "pageviews_zscore", "pageviews_momentum"):
            self.assertIn(col, out.columns)
        self.assertEqual(out["pageviews"].iloc[0], 100)

    def test_placeholder_when_none(self):
        df = self._base_df()
        out = add_pageview_features(df, pageview_df=None)
        self.assertTrue((out["pageviews"] == 0.0).all())
        self.assertIn("pageviews_ma7", out.columns)
        self.assertTrue((out["pageviews_zscore"] == 0.0).all())

    def test_missing_dates_filled_with_zero(self):
        df = self._base_df(5)
        # df の一部の日付のみページビューあり
        partial = pd.DataFrame({"pageviews": [500.0]}, index=[pd.Timestamp("2024-01-03")])
        out = add_pageview_features(df, pageview_df=partial)
        self.assertEqual(out.loc[pd.Timestamp("2024-01-03"), "pageviews"], 500.0)
        self.assertEqual(out.loc[pd.Timestamp("2024-01-01"), "pageviews"], 0.0)

    def test_no_inf_or_nan_in_outputs(self):
        df = self._base_df()
        # 前日 0 → pct_change で inf が出るケース
        views = [0.0, 0.0, 100.0, 0.0, 50.0, 0.0, 0.0, 200.0, 0.0, 0.0]
        pageview_df = pd.DataFrame({"pageviews": views}, index=df.index)
        out = add_pageview_features(df, pageview_df=pageview_df)
        for col in ("pageviews_zscore", "pageviews_momentum"):
            self.assertFalse(np.isinf(out[col]).any(), f"{col} に inf が残存")
            self.assertFalse(out[col].isna().any(), f"{col} に NaN が残存")


if __name__ == "__main__":
    unittest.main()
