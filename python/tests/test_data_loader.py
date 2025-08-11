import unittest
import pandas as pd
from datetime import datetime, timedelta
from python.src.data.data_loader.data_loader import get_stock_data, get_forex_data

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        """
        各テストメソッドの実行前に呼び出されます。
        テストに必要な共通のセットアップを行います。
        """
        self.test_symbol = "AAPL"
        self.test_forex_symbol = "JPY=X"
        # テスト期間を短くし、API呼び出しの頻度を減らす
        self.test_start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        self.test_end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    def test_get_stock_data_returns_dataframe(self):
        """
        get_stock_data関数がpandas.DataFrameを返すことを確認します。
        """
        df = get_stock_data(self.test_symbol, self.test_start_date, self.test_end_date)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty, "Stock data DataFrame should not be empty.")

    def test_get_stock_data_columns(self):
        """
        get_stock_data関数が期待されるカラム（Open, High, Low, Close, Volume）を持つことを確認します。
        """
        df = get_stock_data(self.test_symbol, self.test_start_date, self.test_end_date)
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Column '{col}' not found in stock data.")

    def test_get_stock_data_index_is_datetime(self):
        """
        get_stock_data関数のインデックスがDatetimeIndexであることを確認します。
        """
        df = get_stock_data(self.test_symbol, self.test_start_date, self.test_end_date)
        self.assertIsInstance(df.index, pd.DatetimeIndex, "Index should be a DatetimeIndex.")

    def test_get_stock_data_date_range(self):
        """
        get_stock_data関数が指定された日付範囲のデータをロードすることを確認します。
        """
        df = get_stock_data(self.test_symbol, self.test_start_date, self.test_end_date)
        
        if not df.empty:
            # 取得したデータの最小日付が開始日以降か確認
            self.assertGreaterEqual(df.index.min().date(), datetime.strptime(self.test_start_date, '%Y-%m-%d').date(), "Start date mismatch.")
            # 取得したデータの最大日付が終了日以前か確認
            self.assertLessEqual(df.index.max().date(), datetime.strptime(self.test_end_date, '%Y-%m-%d').date(), "End date mismatch.")
        else:
            # データが空の場合でもテストは成功とする（API側の問題の可能性もあるため）
            self.skipTest("Stock data is empty, skipping date range check.")

    def test_get_stock_data_invalid_symbol(self):
        """
        存在しないシンボルをロードした場合にValueErrorが発生することを確認します。
        """
        with self.assertRaisesRegex(ValueError, "No data found for ticker INVALID_SYMBOL"):
            get_stock_data("INVALID_SYMBOL", self.test_start_date, self.test_end_date)

    def test_get_stock_data_future_date(self):
        """
        未来の日付を指定した場合にValueErrorが発生することを確認します。
        yfinanceは未来の日付を指定してもエラーにならない場合があるため、
        ここではデータが空になることを期待してテストします。
        """
        future_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        # yfinanceの仕様変更により、未来の日付でエラーが発生しない場合があるため、
        # データが空になることを確認するテストに変更
        df = get_stock_data(self.test_symbol, future_date, future_date)
        self.assertTrue(df.empty, "Stock data should be empty for future dates.")

    def test_get_forex_data_returns_dataframe(self):
        """
        get_forex_data関数がpandas.DataFrameを返すことを確認します。
        """
        df = get_forex_data(self.test_forex_symbol, self.test_start_date, self.test_end_date)
        self.assertIsInstance(df, pd.DataFrame, "Forex data should be a DataFrame.")
        self.assertFalse(df.empty, "Forex data DataFrame should not be empty.")

    def test_get_forex_data_invalid_symbol(self):
        """
        存在しない為替シンボルをロードした場合にValueErrorが発生することを確認します。
        """
        with self.assertRaisesRegex(ValueError, "No data found for forex ticker INVALID_FOREX_SYMBOL"):
            get_forex_data("INVALID_FOREX_SYMBOL", self.test_start_date, self.test_end_date)

# if __name__ == '__main__':
#     unittest.main()
