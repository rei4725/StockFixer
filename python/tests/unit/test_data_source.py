"""DataSourceBase / YFinanceDataSource / MockDataSource のユニットテスト"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.market_data.base import DataSourceBase
from src.market_data.yfinance_source import YFinanceDataSource


def _make_ohlcv(periods=60, start="2023-01-01"):
    dates = pd.date_range(start, periods=periods, freq="D")
    return pd.DataFrame(
        {
            "Open": np.linspace(100, 120, periods),
            "High": np.linspace(101, 121, periods),
            "Low": np.linspace(99, 119, periods),
            "Close": np.linspace(100, 120, periods),
            "Volume": np.random.randint(1000, 5000, periods),
        },
        index=dates,
    )


class MockDataSource(DataSourceBase):
    """テスト用モックデータソース。外部 API を一切呼ばない。"""

    def __init__(self, stock_df=None, forex_df=None):
        self._stock_df = stock_df if stock_df is not None else _make_ohlcv(60)
        self._forex_df = forex_df if forex_df is not None else pd.DataFrame(
            {"Close": [150.0]}, index=pd.date_range("2023-01-01", periods=1)
        )
        self.stock_calls: list[tuple] = []
        self.forex_calls: list[tuple] = []

    def get_stock_data(self, symbol, market, start_date, end_date):
        self.stock_calls.append((symbol, market, start_date, end_date))
        return self._stock_df

    def get_forex_data(self, pair, start_date, end_date):
        self.forex_calls.append((pair, start_date, end_date))
        return self._forex_df


class TestDataSourceBase(unittest.TestCase):
    """DataSourceBase の契約テスト"""

    def test_cannot_instantiate_directly(self):
        """抽象クラスは直接インスタンス化できないこと"""
        with self.assertRaises(TypeError):
            DataSourceBase()  # type: ignore[abstract]

    def test_mock_implements_interface(self):
        """MockDataSource が DataSourceBase を正しく実装していること"""
        ds = MockDataSource()
        self.assertIsInstance(ds, DataSourceBase)


class TestMockDataSource(unittest.TestCase):
    """MockDataSource の動作確認"""

    def test_get_stock_data_returns_dataframe(self):
        ds = MockDataSource()
        df = ds.get_stock_data("AAPL", "us", "2023-01-01", "2023-03-31")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_get_stock_data_records_calls(self):
        ds = MockDataSource()
        ds.get_stock_data("AAPL", "us", "2023-01-01", "2023-03-31")
        self.assertEqual(len(ds.stock_calls), 1)
        self.assertEqual(ds.stock_calls[0], ("AAPL", "us", "2023-01-01", "2023-03-31"))

    def test_get_forex_data_returns_dataframe(self):
        ds = MockDataSource()
        df = ds.get_forex_data("JPY=X", "2023-01-01", "2023-03-31")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_get_forex_data_records_calls(self):
        ds = MockDataSource()
        ds.get_forex_data("JPY=X", "2023-01-01", "2023-03-31")
        self.assertEqual(len(ds.forex_calls), 1)
        self.assertEqual(ds.forex_calls[0], ("JPY=X", "2023-01-01", "2023-03-31"))


class TestYFinanceDataSource(unittest.TestCase):
    """YFinanceDataSource の単体テスト（yf_client をモック）"""

    @patch("src.market_data.yfinance_source.yf_client")
    def test_get_stock_data_calls_ticker_history(self, mock_yf_client):
        mock_yf_client.ticker_history.return_value = _make_ohlcv(10)
        ds = YFinanceDataSource()
        result = ds.get_stock_data("AAPL", "us", "2023-01-01", "2023-03-31")
        mock_yf_client.ticker_history.assert_called_once()
        self.assertFalse(result.empty)

    @patch("src.market_data.yfinance_source.yf_client")
    def test_get_stock_data_jp_market_appends_suffix(self, mock_yf_client):
        mock_yf_client.ticker_history.return_value = _make_ohlcv(10)
        ds = YFinanceDataSource()
        ds.get_stock_data("7203", "jp", "2023-01-01", "2023-03-31")
        call_args = mock_yf_client.ticker_history.call_args
        # JP マーケットは ".T" サフィックスが付く
        ticker_used = call_args[0][0]
        self.assertIn(".T", ticker_used)

    @patch("src.market_data.yfinance_source.yf_client")
    def test_get_forex_data_calls_download(self, mock_yf_client):
        mock_yf_client.download.return_value = pd.DataFrame(
            {"Close": [150.0]}, index=pd.date_range("2023-01-01", periods=1)
        )
        ds = YFinanceDataSource()
        result = ds.get_forex_data("JPY=X", "2023-01-01", "2023-03-31")
        mock_yf_client.download.assert_called_once()
        self.assertFalse(result.empty)

    @patch("src.market_data.yfinance_source.yf_client")
    def test_get_forex_data_raises_on_empty(self, mock_yf_client):
        mock_yf_client.download.return_value = pd.DataFrame()
        ds = YFinanceDataSource()
        with self.assertRaises(ValueError):
            ds.get_forex_data("JPY=X", "2023-01-01", "2023-03-31")


class TestPipelineWithMockDataSource(unittest.TestCase):
    """MockDataSource を注入してパイプラインをテストする"""

    def _make_ohlcv_local(self, periods=60):
        return _make_ohlcv(periods)

    @patch("src.market_data.pipeline.fetch_cross_asset_features")
    @patch("src.market_data.pipeline.save_raw_ohlcv")
    @patch("src.market_data.pipeline.should_fetch_fresh_data")
    @patch("src.market_data.pipeline.get_raw_ohlcv_from_db")
    def test_pipeline_uses_mock_data_source_instead_of_real_api(
        self, mock_from_db, mock_fresh_check, mock_save_raw, mock_cross_asset
    ):
        """data_source を注入すると get_stock_data の代わりに使われること"""
        from src.market_data.pipeline import fetch_stock_data_with_features

        mock_from_db.return_value = None
        mock_fresh_check.return_value = True
        mock_cross_asset.return_value = None

        mock_ds = MockDataSource(stock_df=self._make_ohlcv_local(60))
        result = fetch_stock_data_with_features(
            market="us",
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-03-31",
            data_source=mock_ds,
        )

        self.assertIsNotNone(result)
        # MockDataSource が呼ばれたことを確認
        self.assertEqual(len(mock_ds.stock_calls), 1)
        symbol_arg, market_arg, _, _ = mock_ds.stock_calls[0]
        self.assertEqual(symbol_arg, "AAPL")
        self.assertEqual(market_arg, "us")

    @patch("src.market_data.pipeline.fetch_cross_asset_features")
    @patch("src.market_data.pipeline.save_raw_ohlcv")
    @patch("src.market_data.pipeline.should_fetch_fresh_data")
    @patch("src.market_data.pipeline.get_raw_ohlcv_from_db")
    @patch("src.market_data.pipeline.get_stock_data")
    def test_pipeline_without_data_source_calls_default(
        self, mock_get_data, mock_from_db, mock_fresh_check, mock_save_raw, mock_cross_asset
    ):
        """data_source を渡さない場合はデフォルトの get_stock_data が使われること"""
        from src.market_data.pipeline import fetch_stock_data_with_features

        mock_from_db.return_value = None
        mock_fresh_check.return_value = True
        mock_get_data.return_value = self._make_ohlcv_local(60)
        mock_cross_asset.return_value = None

        result = fetch_stock_data_with_features(
            market="us",
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        self.assertIsNotNone(result)
        mock_get_data.assert_called_once()


if __name__ == "__main__":
    unittest.main()
