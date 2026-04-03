"""yf_client モジュールのユニットテスト"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.utils.yf_client import _normalize, download, ticker_history


class TestNormalize(unittest.TestCase):
    """_normalize 関数のテスト"""

    def test_empty_df_returns_empty(self):
        """空 DataFrame をそのまま返すこと"""
        df = pd.DataFrame()
        result = _normalize(df)
        self.assertTrue(result.empty)

    def test_multiindex_columns_flattened_to_level0(self):
        """MultiIndex カラムが第 1 レベル（フィールド名）のみに落とされること"""
        multi_idx = pd.MultiIndex.from_tuples(
            [("Close", "AAPL"), ("Open", "AAPL"), ("Volume", "AAPL")]
        )
        df = pd.DataFrame(
            [[100.0, 99.0, 1000]],
            columns=multi_idx,
            index=pd.date_range("2024-01-01", periods=1),
        )
        result = _normalize(df)

        self.assertFalse(isinstance(result.columns, pd.MultiIndex))
        self.assertIn("Close", result.columns)
        self.assertIn("Open", result.columns)

    def test_all_nan_rows_removed(self):
        """全カラムが NaN の行が除去されること"""
        df = pd.DataFrame(
            {"Close": [100.0, float("nan")], "Open": [99.0, float("nan")]},
            index=pd.date_range("2024-01-01", periods=2),
        )
        result = _normalize(df)
        self.assertEqual(len(result), 1)

    def test_partial_nan_rows_kept(self):
        """一部カラムのみ NaN の行は保持されること（全 NaN のみ除去）"""
        df = pd.DataFrame(
            {"Close": [100.0, 101.0], "Open": [99.0, float("nan")]},
            index=pd.date_range("2024-01-01", periods=2),
        )
        result = _normalize(df)
        self.assertEqual(len(result), 2)

    def test_tz_aware_index_becomes_naive(self):
        """タイムゾーン付きインデックスが tz-naive に変換されること"""
        idx = pd.date_range("2024-01-01", periods=3, tz="UTC")
        df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=idx)

        result = _normalize(df)

        self.assertIsNone(result.index.tzinfo)

    def test_tz_naive_index_unchanged(self):
        """tz-naive インデックスはそのまま保持されること"""
        idx = pd.date_range("2024-01-01", periods=3)
        df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=idx)

        result = _normalize(df)

        self.assertIsNone(result.index.tzinfo)
        self.assertEqual(len(result), 3)

    def test_us_eastern_tz_removed(self):
        """US/Eastern などのタイムゾーンも除去されること"""
        idx = pd.date_range("2024-01-01", periods=2, tz="US/Eastern")
        df = pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)

        result = _normalize(df)

        self.assertIsNone(result.index.tzinfo)

    def test_normal_df_unchanged(self):
        """通常の（MultiIndex なし・tz-naive・NaN なし）DataFrame はそのまま返ること"""
        df = pd.DataFrame(
            {"Close": [100.0, 101.0], "Volume": [1000, 2000]},
            index=pd.date_range("2024-01-01", periods=2),
        )
        result = _normalize(df)
        self.assertEqual(list(result.columns), ["Close", "Volume"])
        self.assertEqual(len(result), 2)


class TestDownload(unittest.TestCase):
    """download 関数のテスト"""

    @patch("src.utils.yf_client.with_retry")
    def test_period_mode_returns_normalized_df(self, mock_retry):
        """period 指定で正規化済み DataFrame が返ること"""
        mock_df = pd.DataFrame(
            {"Close": [150.0], "Open": [149.0]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        mock_retry.return_value = mock_df

        result = download("AAPL", period="1d")

        self.assertFalse(result.empty)
        self.assertIn("Close", result.columns)

    @patch("src.utils.yf_client.with_retry")
    def test_start_end_mode_returns_normalized_df(self, mock_retry):
        """start/end 指定で正規化済み DataFrame が返ること"""
        mock_df = pd.DataFrame(
            {"Close": [200.0]},
            index=pd.date_range("2024-06-01", periods=1),
        )
        mock_retry.return_value = mock_df

        result = download("MSFT", start="2024-06-01", end="2024-07-01")

        self.assertFalse(result.empty)

    @patch("src.utils.yf_client.with_retry")
    def test_exception_returns_empty_df(self, mock_retry):
        """例外発生時は空 DataFrame が返ること"""
        mock_retry.side_effect = Exception("network error")

        result = download("AAPL", period="1d")

        self.assertTrue(result.empty)
        self.assertIsInstance(result, pd.DataFrame)

    @patch("src.utils.yf_client.with_retry")
    def test_multiindex_result_normalized(self, mock_retry):
        """MultiIndex カラムを返す場合も正規化されること"""
        multi_idx = pd.MultiIndex.from_tuples([("Close", "AAPL"), ("Open", "AAPL")])
        mock_df = pd.DataFrame(
            [[100.0, 99.0]],
            columns=multi_idx,
            index=pd.date_range("2024-01-01", periods=1),
        )
        mock_retry.return_value = mock_df

        result = download("AAPL", period="1d")

        self.assertFalse(isinstance(result.columns, pd.MultiIndex))

    @patch("src.utils.yf_client.with_retry")
    def test_tz_aware_result_normalized(self, mock_retry):
        """tz-aware インデックスが tz-naive に変換されること"""
        idx = pd.date_range("2024-01-01", periods=2, tz="UTC")
        mock_df = pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)
        mock_retry.return_value = mock_df

        result = download("7203.T", start="2024-01-01", end="2024-01-03")

        self.assertIsNone(result.index.tzinfo)

    @patch("src.utils.yf_client.with_retry")
    def test_empty_df_from_yfinance_returns_empty(self, mock_retry):
        """yfinance が空 DataFrame を返した場合は空のまま返ること"""
        mock_retry.return_value = pd.DataFrame()

        result = download("UNKNOWN", period="1d")

        self.assertTrue(result.empty)


class TestTickerHistory(unittest.TestCase):
    """ticker_history 関数のテスト"""

    @patch("src.utils.yf_client.with_retry")
    @patch("src.utils.yf_client.yf")
    def test_returns_normalized_df(self, mock_yf, mock_retry):
        """正常時は正規化済み DataFrame が返ること"""
        mock_yf.Ticker.return_value = MagicMock()
        mock_df = pd.DataFrame(
            {"Close": [200.0], "Open": [199.0]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        mock_retry.return_value = mock_df

        result = ticker_history("7203.T", start="2024-01-01", end="2024-06-01")

        self.assertFalse(result.empty)
        self.assertIn("Close", result.columns)

    @patch("src.utils.yf_client.with_retry")
    @patch("src.utils.yf_client.yf")
    def test_exception_returns_empty(self, mock_yf, mock_retry):
        """例外発生時は空 DataFrame が返ること"""
        mock_yf.Ticker.return_value = MagicMock()
        mock_retry.side_effect = Exception("timeout")

        result = ticker_history("7203.T", start="2024-01-01", end="2024-06-01")

        self.assertTrue(result.empty)
        self.assertIsInstance(result, pd.DataFrame)

    @patch("src.utils.yf_client.with_retry")
    @patch("src.utils.yf_client.yf")
    def test_ticker_object_created(self, mock_yf, mock_retry):
        """Ticker オブジェクトが正しいシンボルで生成されること"""
        mock_yf.Ticker.return_value = MagicMock()
        mock_retry.return_value = pd.DataFrame(
            {"Close": [100.0]}, index=pd.date_range("2024-01-01", periods=1)
        )

        ticker_history("AAPL", start="2024-01-01", end="2024-06-01")

        mock_yf.Ticker.assert_called_once_with("AAPL")

    @patch("src.utils.yf_client.with_retry")
    @patch("src.utils.yf_client.yf")
    def test_tz_aware_index_normalized(self, mock_yf, mock_retry):
        """tz-aware なインデックスが tz-naive に変換されること"""
        mock_yf.Ticker.return_value = MagicMock()
        idx = pd.date_range("2024-01-01", periods=3, tz="Asia/Tokyo")
        mock_retry.return_value = pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=idx)

        result = ticker_history("7203.T", start="2024-01-01", end="2024-06-01")

        self.assertIsNone(result.index.tzinfo)

    @patch("src.utils.yf_client.with_retry")
    @patch("src.utils.yf_client.yf")
    def test_empty_df_from_yfinance(self, mock_yf, mock_retry):
        """yfinance が空 DataFrame を返した場合は空のまま返ること"""
        mock_yf.Ticker.return_value = MagicMock()
        mock_retry.return_value = pd.DataFrame()

        result = ticker_history("UNKNOWN", start="2024-01-01", end="2024-01-02")

        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
