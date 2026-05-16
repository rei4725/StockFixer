import os
import tempfile
import unittest
from datetime import datetime, timedelta

import pandas as pd

import src.utils.data_path_utils as path_utils
import src.utils.db as db_module
from src.market_data.loader import (
    get_forex_data,
    get_stock_data,
    get_stock_data_auto,
    get_stock_data_from_file,
)


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        """
        各テストメソッドの実行前に呼び出されます。
        テストに必要な共通のセットアップを行います。
        """
        self.test_symbol = "AAPL"
        self.test_forex_symbol = "JPY=X"
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
            os.rmdir(self.tmp_dir)

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

    def test_get_forex_data_returns_dataframe(self):
        """get_forex_data関数がpandas.DataFrameを返すことを確認します。"""
        df = get_forex_data(self.test_forex_symbol, self.test_start_date, self.test_end_date)
        self.assertIsInstance(df, pd.DataFrame, "Forex data should be a DataFrame.")
        self.assertFalse(df.empty, "Forex data DataFrame should not be empty.")

    def test_get_forex_data_invalid_symbol(self):
        """存在しない為替シンボルをロードした場合に例外が発生することを確認します。"""
        try:
            get_forex_data("INVALID_FOREX_SYMBOL", self.test_start_date, self.test_end_date)
            self.fail("例外が発生しませんでした")
        except Exception as e:
            self.assertTrue(isinstance(e, ValueError) or "YFTzMissingError" in str(e))

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

    def test_get_forex_data_columns_and_index(self):
        """get_forex_data関数が期待されるカラム（Open, High, Low, Close, Volume）とDatetimeIndexを持つことを確認します。"""
        df = get_forex_data(self.test_forex_symbol, self.test_start_date, self.test_end_date)
        expected_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Column '{col}' not found in forex data.")
        self.assertIsInstance(df.index, pd.DatetimeIndex, "Index should be a DatetimeIndex.")

    def test_get_stock_data_empty_type(self):
        """データが空の場合でもDataFrame型が返ることを確認します。"""
        df = get_stock_data("us", self.test_symbol, "1900-01-01", "1900-01-02")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_get_forex_data_empty_type(self):
        """為替データが空の場合でもDataFrame型が返ることを確認します。"""
        try:
            df = get_forex_data(self.test_forex_symbol, "1900-01-01", "1900-01-02")
            self.assertIsInstance(df, pd.DataFrame)
            self.assertTrue(df.empty)
        except Exception as e:
            self.assertTrue(
                isinstance(e, ValueError)
                or "YFPricesMissingError" in str(e)
                or "possibly delisted" in str(e)
            )

    def test_get_stock_data_from_file(self):
        """get_stock_data_from_fileがDBから正しくデータを取得できることを確認します。"""
        from src.utils.db import upsert_stock_features

        market = "us"
        symbol = "TEST"
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2022-01-01", periods=3).astype(str),
                "Open": [1.0, 2.0, 3.0],
                "High": [2.0, 3.0, 4.0],
                "Low": [0.0, 1.0, 2.0],
                "Close": [1.5, 2.5, 3.5],
                "Volume": [100, 200, 300],
            }
        )
        upsert_stock_features(market, symbol, df)
        result = get_stock_data_from_file(market, symbol, "2022-01-01", "2022-01-03")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)

    def test_get_stock_data_from_file_no_csv(self):
        """DBにデータが存在しない場合にFileNotFoundErrorが発生することを確認します。"""
        market = "us"
        symbol = "NOFILE"
        with self.assertRaises(FileNotFoundError):
            get_stock_data_from_file(market, symbol, "2022-01-01", "2022-01-03")

    def test_get_stock_data_auto_file(self):
        """get_stock_data_autoがsource='file'でDBからデータを取得することを確認します。"""
        from src.utils.db import upsert_stock_features

        market = "us"
        symbol = "AUTO"
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2022-01-01", periods=2).astype(str),
                "Open": [1.0, 2.0],
                "High": [2.0, 3.0],
                "Low": [0.0, 1.0],
                "Close": [1.5, 2.5],
                "Volume": [100, 200],
            }
        )
        upsert_stock_features(market, symbol, df)
        result = get_stock_data_auto(market, symbol, "2022-01-01", "2022-01-02", source="file")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

    def test_get_stock_data_auto_api(self):
        """get_stock_data_autoがsource='api'でget_stock_dataを呼ぶことを確認します。"""
        df = get_stock_data_auto(
            "", self.test_symbol, self.test_start_date, self.test_end_date, source="api"
        )
        self.assertIsInstance(df, pd.DataFrame)

    def test_get_stock_data_auto_invalid_source(self):
        """get_stock_data_autoのsource引数が不正な場合にValueErrorが発生することを確認します。"""
        with self.assertRaises(ValueError):
            get_stock_data_auto(
                "us", "AAPL", self.test_start_date, self.test_end_date, source="invalid"
            )


# if __name__ == '__main__':
#     unittest.main()
