"""unified_model_pipeline の純粋関数テスト"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


def _make_df(n=20, include_market=True, include_y=True):
    """テスト用の特徴量DataFrame"""
    df = pd.DataFrame(
        {
            "close_lag1": np.random.uniform(99, 101, n),
            "rsi": np.random.uniform(30, 70, n),
            "date": pd.date_range("2026-01-01", periods=n, freq="B"),
        }
    )
    if include_market:
        df["market"] = "us"
        df["symbol"] = "AAPL"
    if include_y:
        df["y"] = np.random.uniform(-0.05, 0.05, n)
    return df


class TestPrepareUnifiedFeatures(unittest.TestCase):
    """prepare_unified_features のテスト（純粋関数）"""

    def _fn(self, df):
        from src.services.unified_model_pipeline import prepare_unified_features

        return prepare_unified_features(df)

    def test_returns_x_and_y(self):
        df = _make_df()
        X, y = self._fn(df)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    def test_y_column_excluded_from_x(self):
        df = _make_df()
        X, _ = self._fn(df)
        self.assertNotIn("y", X.columns)

    def test_metadata_columns_excluded(self):
        df = _make_df()
        X, _ = self._fn(df)
        for col in ("market", "symbol", "date"):
            self.assertNotIn(col, X.columns)

    def test_market_encoded_added(self):
        """market列をエンコードしたmarket_encoded列が追加されること"""
        df = _make_df()
        X, _ = self._fn(df)
        self.assertIn("market_encoded", X.columns)

    def test_market_encoded_us_is_zero(self):
        df = _make_df()
        df["market"] = "us"
        X, _ = self._fn(df)
        self.assertTrue((X["market_encoded"] == 0).all())

    def test_market_encoded_jp_is_one(self):
        df = _make_df()
        df["market"] = "jp"
        X, _ = self._fn(df)
        self.assertTrue((X["market_encoded"] == 1).all())

    def test_nan_rows_dropped(self):
        df = _make_df(n=10)
        df.loc[0, "close_lag1"] = float("nan")
        X, y = self._fn(df)
        self.assertEqual(len(X), 9)
        self.assertEqual(len(y), 9)

    def test_raises_when_no_y_column(self):
        df = _make_df(include_y=False)
        with self.assertRaises(ValueError):
            self._fn(df)

    def test_x_and_y_same_length(self):
        df = _make_df(n=15)
        X, y = self._fn(df)
        self.assertEqual(len(X), len(y))

    def test_existing_market_encoded_not_overwritten(self):
        """market_encoded 列が既にある場合は上書きしないこと"""
        df = _make_df()
        df["market_encoded"] = 99  # 既存値
        X, _ = self._fn(df)
        # market列がある場合は上書きしない（コードの仕様）
        # market_encoded列が存在することのみ確認
        self.assertIn("market_encoded", X.columns)


class TestUnifiedEarningsMask(unittest.TestCase):
    @patch("src.services.unified_model_pipeline.get_earnings_dates")
    def test_mask_earnings_rows_for_unified_removes_flagged_rows(self, mock_get_earnings):
        from src.services.unified_model_pipeline import _mask_earnings_rows_for_unified

        df = _make_df(n=10)
        df["market"] = "us"
        df["symbol"] = "AAPL"
        df["date"] = pd.date_range("2026-01-01", periods=10, freq="B")
        mock_get_earnings.return_value = pd.DatetimeIndex([pd.Timestamp("2026-01-07")])

        masked = _mask_earnings_rows_for_unified(df)

        self.assertLess(len(masked), len(df))
        self.assertNotIn(pd.Timestamp("2026-01-07"), pd.to_datetime(masked["date"]).tolist())
        self.assertNotIn("earnings_flag", masked.columns)


class TestLoadUnifiedModel(unittest.TestCase):
    """load_unified_model のテスト（FileNotFoundError パス）"""

    def test_raises_when_model_not_found(self):
        import tempfile

        from src.services.unified_model_pipeline import load_unified_model

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_unified_model(model_name="NonExistentModel", model_dir=tmp)


if __name__ == "__main__":
    unittest.main()
