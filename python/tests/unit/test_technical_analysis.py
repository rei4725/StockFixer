import unittest

import numpy as np
import pandas as pd

from src.features import technical_analysis
from src.features.market_regime import get_market_regime


class TestTechnicalAnalysis(unittest.TestCase):
    def setUp(self):
        # OHLCVデータ（Open, High, Low, Close, Volume）の基本テスト用DataFrame
        # カラム型も float（価格）・int（出来高）で揃える
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        self.df = pd.DataFrame(
            {
                "Open": np.linspace(100, 120, 20).astype(float),
                "High": np.linspace(101, 121, 20).astype(float),
                "Low": np.linspace(99, 119, 20).astype(float),
                "Close": np.linspace(100, 120, 20).astype(float),
                "Volume": np.random.randint(1000, 2000, 20).astype(int),
            },
            index=dates,
        )

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
        expected_y = ((self.df["Close"].shift(-1) - self.df["Close"]) / self.df["Close"]).reindex(
            y.index
        )
        pd.testing.assert_series_equal(y, expected_y, check_names=False)

    def test_create_basic_lag_features_with_feature_cols(self):
        X, y = technical_analysis.create_basic_lag_features(
            self.df, n_lags=3, feature_cols=["Open", "Close"]
        )
        self.assertEqual(X.shape[1], 2 * 3)
        self.assertIn("Open_lag1", X.columns)
        self.assertIn("Close_lag3", X.columns)

    def test_add_technical_indicators(self):
        """OHLCVデータにテクニカル指標・季節性特徴量が追加されることを確認する。"""
        df_with_ind = technical_analysis.add_technical_indicators(self.df.copy())
        expected_cols = [
            "rsi",
            "macd",
            "macd_signal",
            "macd_diff",
            "ema_fast",
            "ema_slow",
            "atr",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "bb_width",
            "stoch_k",
            "stoch_d",
            "obv",
            "day_of_week",
            "month",
            "is_month_end",
        ]
        for col in expected_cols:
            self.assertIn(col, df_with_ind.columns)
        # RSI値の範囲（NaNは無視して判定）
        rsi_no_nan = df_with_ind["rsi"].dropna()
        self.assertTrue(((rsi_no_nan >= 0) & (rsi_no_nan <= 100)).all())
        # 季節性特徴量の値範囲確認
        self.assertTrue(df_with_ind["day_of_week"].between(0, 6).all())
        self.assertTrue(df_with_ind["month"].between(1, 12).all())
        self.assertTrue(df_with_ind["is_month_end"].isin([0, 1]).all())


class TestMarketRegime(unittest.TestCase):
    def test_get_market_regime_identifies_bull_market(self):
        dates = pd.date_range("2024-01-01", periods=8, freq="D")
        df = pd.DataFrame(
            {
                "High": [101, 102, 103, 106, 108, 106, 107, 108],
                "Low": [99, 100, 101, 100, 99, 104, 105, 106],
                "Close": [100, 101, 102, 103, 104, 105, 106, 107],
            },
            index=dates,
        )

        regime = get_market_regime(df, ema_window=3, atr_window=2, vol_high_quantile=0.95)
        self.assertEqual(regime.iloc[-1], "bull")

    def test_get_market_regime_identifies_bear_market(self):
        dates = pd.date_range("2024-01-01", periods=8, freq="D")
        df = pd.DataFrame(
            {
                "High": [108, 107, 110, 112, 104, 103, 102, 101],
                "Low": [106, 105, 97, 95, 102, 101, 100, 99],
                "Close": [107, 106, 105, 104, 103, 102, 101, 100],
            },
            index=dates,
        )

        regime = get_market_regime(df, ema_window=3, atr_window=2, vol_high_quantile=0.95)
        self.assertEqual(regime.iloc[-1], "bear")

    def test_get_market_regime_identifies_range_market_on_high_volatility(self):
        dates = pd.date_range("2024-01-01", periods=8, freq="D")
        df = pd.DataFrame(
            {
                "High": [101, 102, 103, 104, 105, 110, 120, 135],
                "Low": [99, 100, 101, 102, 103, 96, 90, 80],
                "Close": [100, 101, 102, 103, 104, 105, 106, 107],
            },
            index=dates,
        )

        regime = get_market_regime(df, ema_window=3, atr_window=2, vol_high_quantile=0.5)
        self.assertEqual(regime.iloc[-1], "range")

    def test_classify_regime_remains_compatible_wrapper(self):
        dates = pd.date_range("2024-01-01", periods=6, freq="D")
        df = pd.DataFrame(
            {
                "High": [101, 102, 103, 104, 105, 106],
                "Low": [99, 100, 101, 102, 103, 104],
                "Close": [100, 101, 102, 103, 104, 105],
            },
            index=dates,
        )

        wrapped = technical_analysis.classify_regime(df, ema_window=3, atr_window=2)
        direct = get_market_regime(df, ema_window=3, atr_window=2)
        pd.testing.assert_series_equal(wrapped, direct)


if __name__ == "__main__":
    unittest.main()
