import unittest
import pandas as pd
import numpy as np
from src.features import technical_analysis

class TestTechnicalAnalysis(unittest.TestCase):

    def setUp(self):
        # run_data_creation.py で add_technical_indicators に渡される df と同じ形式（OHLCV: Open, High, Low, Close, Volume）
        # カラム型も float（価格）・int（出来高）で揃える
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        self.df = pd.DataFrame({
            'Open': np.linspace(100, 120, 20).astype(float),
            'High': np.linspace(101, 121, 20).astype(float),
            'Low': np.linspace(99, 119, 20).astype(float),
            'Close': np.linspace(100, 120, 20).astype(float),
            'Volume': np.random.randint(1000, 2000, 20).astype(int)
        }, index=dates)

    def test_create_basic_lag_features_default(self):
        X, y = technical_analysis.create_basic_lag_features(self.df)
        n_lags = 5
        num_features = len(self.df.select_dtypes(include=[float, int]).columns)
        self.assertEqual(X.shape[1], num_features * n_lags)
        self.assertEqual(len(X), len(y))
        self.assertFalse(X.isnull().any().any())
        self.assertFalse(y.isnull().any())
        # yは翌日の変化率（インデックスが揃っていることのみ確認）
        self.assertTrue((y.index == X.index).all())
        # yの値が元データの変化率と一致する（インデックスで比較）
        expected_y = ((self.df['Close'].shift(-1) - self.df['Close']) / self.df['Close']).reindex(y.index)
        pd.testing.assert_series_equal(y, expected_y, check_names=False)

    def test_create_basic_lag_features_with_feature_cols(self):
        X, y = technical_analysis.create_basic_lag_features(self.df, n_lags=3, feature_cols=['Open', 'Close'])
        self.assertEqual(X.shape[1], 2 * 3)
        self.assertIn('Open_lag1', X.columns)
        self.assertIn('Close_lag3', X.columns)

    def test_add_technical_indicators(self):
        """
        run_data_creation.py で add_technical_indicators に渡される df（OHLCV形式）と同じ構造のデータでテストする
        """
        df_with_ind = technical_analysis.add_technical_indicators(self.df.copy())
        for col in ['rsi', 'macd', 'macd_signal', 'macd_diff', 'ema_fast', 'ema_slow', 'atr']:
            self.assertIn(col, df_with_ind.columns)
        # RSI値の範囲（NaNは無視して判定）
        rsi_no_nan = df_with_ind['rsi'].dropna()
        self.assertTrue(((rsi_no_nan >= 0) & (rsi_no_nan <= 100)).all())

if __name__ == '__main__':
    unittest.main()
