"""ユニットテスト: src.trading.regime_leverage_strategy.indicators"""

import unittest

import numpy as np
import pandas as pd

from src.trading.regime_leverage_strategy.indicators import build_weekly_frame, wilder_atr


class TestWilderAtr(unittest.TestCase):
    def test_atr_is_nan_before_period(self):
        idx = pd.bdate_range("2026-01-01", periods=20)
        df = pd.DataFrame(
            {
                "High": np.full(20, 105.0),
                "Low": np.full(20, 95.0),
                "Close": np.full(20, 100.0),
            },
            index=idx,
        )
        atr = wilder_atr(df, period=14)
        self.assertTrue(atr.iloc[:13].isna().all())
        self.assertFalse(pd.isna(atr.iloc[13]))

    def test_atr_reflects_true_range(self):
        idx = pd.bdate_range("2026-01-01", periods=20)
        df = pd.DataFrame(
            {
                "High": np.full(20, 110.0),
                "Low": np.full(20, 90.0),
                "Close": np.full(20, 100.0),
            },
            index=idx,
        )
        atr = wilder_atr(df, period=14)
        # High-Low=20 が唯一の候補(前日終値との差より大きい)なのでATRは20に収束する
        self.assertAlmostEqual(atr.iloc[-1], 20.0, places=6)


class TestBuildWeeklyFrame(unittest.TestCase):
    def test_adds_ma200_and_week_close(self):
        idx = pd.bdate_range("2024-01-01", periods=260)
        rng = np.random.default_rng(0)
        prices = 100 + np.cumsum(rng.normal(0, 1, 260))
        df = pd.DataFrame(
            {
                "Open": prices,
                "High": prices + 1,
                "Low": prices - 1,
                "Close": prices,
                "Volume": np.full(260, 1_000_000),
            },
            index=idx,
        )
        result = build_weekly_frame(df)
        self.assertIn("MA200", result.columns)
        self.assertIn("ATR14", result.columns)
        # 200本目以降はMA200が非NaN
        self.assertFalse(pd.isna(result["MA200"].iloc[-1]))


if __name__ == "__main__":
    unittest.main()
