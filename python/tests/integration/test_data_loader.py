import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

import pandas as pd

import src.utils.data_path_utils as path_utils
import src.utils.db as db_module
from src.market_data.loader import get_stock_data


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        """
        各テストメソッドの実行前に呼び出されます。
        テストに必要な共通のセットアップを行います。
        """
        self.test_symbol = "AAPL"
        # テスト期間を短くし、API呼び出しの頻度を減らす
        self.test_start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        self.test_end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        # DB関連テスト用: テスト用一時DBを設定
        db_module.close_connection()
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_db = os.path.join(self.tmp_dir, "test_loader.duckdb")
        self._orig_get_db_path = path_utils.get_db_path
        path_utils.get_db_path = lambda: self.tmp_db
        db_module.get_db_path = lambda: self.tmp_db
        db_module._tables_initialized = False

    def tearDown(self):
        """DB接続のクリーンアップ"""
        db_module.close_connection()
        path_utils.get_db_path = self._orig_get_db_path
        db_module.get_db_path = self._orig_get_db_path
        if os.path.exists(self.tmp_db):
            os.remove(self.tmp_db)
        wal_path = self.tmp_db + ".wal"
        if os.path.exists(wal_path):
            os.remove(wal_path)
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_get_stock_data_returns_dataframe(self):
        """get_stock_data関数がpandas.DataFrameを返すことを確認します。"""
        df = get_stock_data("us", self.test_symbol, self.test_start_date, self.test_end_date)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty, "Stock data DataFrame should not be empty.")

    def test_get_stock_data_jp_dis(self):
        """日本株ディスコ（6146）の株価データが取得できることを確認"""
        df = get_stock_data("jp", "6146", "2024-01-01", "2024-01-31")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty, "日本株ディスコの株価データが取得できませんでした。")

    def test_get_stock_data_columns(self):
        """get_stock_data関数が期待されるカラム（Open, High, Low, Close, Volume）を持つことを確認します。"""
        df = get_stock_data("us", self.test_symbol, self.test_start_date, self.test_end_date)
        expected_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Column '{col}' not found in stock data.")

    def test_get_stock_data_index_is_datetime(self):
        """get_stock_data関数のインデックスがDatetimeIndexであることを確認します。"""
        df = get_stock_data("us", self.test_symbol, self.test_start_date, self.test_end_date)
        self.assertIsInstance(df.index, pd.DatetimeIndex, "Index should be a DatetimeIndex.")

    def test_get_stock_data_date_range(self):
        """get_stock_data関数が指定された日付範囲のデータをロードすることを確認します。"""
        df = get_stock_data("us", self.test_symbol, self.test_start_date, self.test_end_date)

        if not df.empty:
            # 取得したデータの最小日付が開始日以降か確認
            self.assertGreaterEqual(
                df.index.min().date(),
                datetime.strptime(self.test_start_date, "%Y-%m-%d").date(),
                "Start date mismatch.",
            )
            # 取得したデータの最大日付が終了日以前か確認
            self.assertLessEqual(
                df.index.max().date(),
                datetime.strptime(self.test_end_date, "%Y-%m-%d").date(),
                "End date mismatch.",
            )
        else:
            # データが空の場合でもテストは成功とする（API側の問題の可能性もあるため）
            self.skipTest("Stock data is empty, skipping date range check.")

    def test_get_stock_data_invalid_symbol(self):
        """存在しないシンボルをロードした場合にValueErrorが発生するか、空DataFrameが返ることを確認します。"""
        try:
            df = get_stock_data("us", "INVALID_SYMBOL", self.test_start_date, self.test_end_date)
            # 空DataFrameならOK
            self.assertTrue(df.empty or "No data found" in str(df))
        except Exception as e:
            self.assertTrue(
                isinstance(e, ValueError) or "No data found for ticker INVALID_SYMBOL" in str(e)
            )

    def test_get_stock_data_future_date(self):
        """
        未来の日付を指定した場合にValueErrorが発生することを確認します。
        yfinanceは未来の日付を指定してもエラーにならない場合があるため、
        ここではデータが空になることを期待してテストします。
        """
        future_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        # yfinanceの仕様変更により、未来の日付でエラーが発生しない場合があるため、
        # データが空になることを確認するテストに変更
        df = get_stock_data("us", self.test_symbol, future_date, future_date)
        self.assertTrue(df.empty, "Stock data should be empty for future dates.")

    def test_get_stock_data_start_after_end(self):
        """開始日が終了日より後の場合にValueErrorが発生するか、空DataFrameが返ることを確認します。"""
        try:
            df = get_stock_data("us", self.test_symbol, self.test_end_date, self.test_start_date)
            self.assertTrue(df.empty or "Invalid input" in str(df))
        except Exception as e:
            self.assertTrue(isinstance(e, ValueError) or "Invalid input" in str(e))

    def test_get_stock_data_invalid_date_format(self):
        """日付フォーマットが不正な場合にValueErrorが発生するか、空DataFrameが返ることを確認します。"""
        try:
            df = get_stock_data("us", self.test_symbol, "2025/01/01", self.test_end_date)
            self.assertTrue(df.empty or "does not match format" in str(df))
        except Exception as e:
            self.assertTrue(isinstance(e, ValueError) or "does not match format" in str(e))

    def test_get_stock_data_empty_type(self):
        """データが空の場合でもDataFrame型が返ることを確認します。"""
        df = get_stock_data("us", self.test_symbol, "1900-01-01", "1900-01-02")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)


# if __name__ == '__main__':
#     unittest.main()
