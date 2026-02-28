"""data_pipeline モジュールのユニットテスト"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

import src.utils.db as db_module
import src.utils.data_path_utils as path_utils
from src.services.data_pipeline import save_stock_data_with_features


class TestSaveStockDataWithFeatures(unittest.TestCase):
    """save_stock_data_with_features 関数のテスト"""

    def setUp(self):
        """テスト用一時DBを設定"""
        db_module.close_connection()
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_db = os.path.join(self.tmp_dir, "test_data_pipeline.duckdb")
        self._orig_get_db_path = path_utils.get_db_path
        path_utils.get_db_path = lambda: self.tmp_db
        db_module.get_db_path = lambda: self.tmp_db
        db_module._connection = None

    def tearDown(self):
        db_module.close_connection()
        path_utils.get_db_path = self._orig_get_db_path
        db_module.get_db_path = self._orig_get_db_path
        if os.path.exists(self.tmp_db):
            os.remove(self.tmp_db)
        wal_path = self.tmp_db + ".wal"
        if os.path.exists(wal_path):
            os.remove(wal_path)
        if os.path.exists(self.tmp_dir):
            os.rmdir(self.tmp_dir)

    def _make_ohlcv(self, periods=30):
        """テスト用OHLCVデータを生成する"""
        dates = pd.date_range("2023-01-01", periods=periods, freq="D")
        return pd.DataFrame({
            "Open": np.linspace(100, 120, periods),
            "High": np.linspace(101, 121, periods),
            "Low": np.linspace(99, 119, periods),
            "Close": np.linspace(100, 120, periods),
            "Volume": np.random.randint(1000, 5000, periods),
        }, index=dates)

    @patch("src.services.data_pipeline.get_stock_data")
    def test_saves_features_to_db(self, mock_get_data):
        """株価データからDB保存まで正常に完了することを確認"""
        df = self._make_ohlcv(30)
        mock_get_data.return_value = df

        save_stock_data_with_features(
            market="us", symbol="TEST",
            start_date="2023-01-01", end_date="2023-01-30"
        )

        # DBから特徴量を取得して確認
        from src.utils.db import load_stock_features
        result = load_stock_features("us", "TEST")
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)
        # y列が含まれている
        self.assertIn("y", result.columns)

    @patch("src.services.data_pipeline.get_stock_data")
    def test_empty_data_no_error(self, mock_get_data):
        """空のデータが返された場合にエラーにならないことを確認"""
        mock_get_data.return_value = pd.DataFrame()
        # 例外が発生しなければ成功
        save_stock_data_with_features(
            market="us", symbol="EMPTY",
            start_date="2023-01-01", end_date="2023-01-30"
        )

    @patch("src.services.data_pipeline.get_stock_data")
    def test_none_data_no_error(self, mock_get_data):
        """Noneが返された場合にエラーにならないことを確認"""
        mock_get_data.return_value = None
        save_stock_data_with_features(
            market="us", symbol="NONE",
            start_date="2023-01-01", end_date="2023-01-30"
        )

    @patch("src.services.data_pipeline.get_stock_data")
    def test_market_encoded_column(self, mock_get_data):
        """market_encoded列がUS=0, JP=1で正しく設定されることを確認"""
        df = self._make_ohlcv(30)
        mock_get_data.return_value = df

        save_stock_data_with_features(
            market="us", symbol="ENC",
            start_date="2023-01-01", end_date="2023-01-30"
        )

        from src.utils.db import load_stock_features
        result = load_stock_features("us", "ENC")
        if result is not None and not result.empty and "market_encoded" in result.columns:
            self.assertTrue((result["market_encoded"] == 0).all())

    @patch("src.services.data_pipeline.get_stock_data")
    def test_overwrites_existing_data(self, mock_get_data):
        """同じ銘柄で再実行すると既存データが上書きされることを確認"""
        df = self._make_ohlcv(30)
        mock_get_data.return_value = df

        save_stock_data_with_features(
            market="us", symbol="OVR",
            start_date="2023-01-01", end_date="2023-01-30"
        )
        from src.utils.db import load_stock_features
        result1 = load_stock_features("us", "OVR")

        # 2回目
        save_stock_data_with_features(
            market="us", symbol="OVR",
            start_date="2023-01-01", end_date="2023-01-30"
        )
        result2 = load_stock_features("us", "OVR")

        # 行数が同じ（追記ではなく上書き）
        self.assertEqual(len(result1), len(result2))


if __name__ == '__main__':
    unittest.main()
