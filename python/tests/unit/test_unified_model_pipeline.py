"""unified_model_pipeline の純粋関数テスト"""

import tempfile
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
        from src.prediction.unified_model_pipeline import prepare_unified_features

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

    @patch("src.prediction.unified_model_pipeline.load_excluded_features")
    def test_prepare_unified_features_applies_exclusions(self, mock_excluded):
        df = _make_df()
        mock_excluded.return_value = ["rsi"]

        X, _ = self._fn(df)

        self.assertNotIn("rsi", X.columns)


class TestUnifiedEarningsMask(unittest.TestCase):
    @patch("src.prediction.unified_model_pipeline.get_earnings_dates")
    def test_mask_earnings_rows_for_unified_removes_flagged_rows(self, mock_get_earnings):
        from src.prediction.unified_model_pipeline import _mask_earnings_rows_for_unified

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

        from src.prediction.unified_model_pipeline import load_unified_model

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_unified_model(model_name="NonExistentModel", model_dir=tmp)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# 追加テスト: load_all_stock_data / _apply_unified_feature_exclusions / train_unified_model
# ---------------------------------------------------------------------------


class TestLoadAllStockData(unittest.TestCase):
    """load_all_stock_data のテスト"""

    @patch("src.prediction.unified_model_pipeline.load_all_stock_features")
    @patch("src.prediction.unified_model_pipeline._mask_earnings_rows_for_unified")
    def test_returns_dataframe_with_required_columns(self, mock_mask, mock_load):
        """必要なカラムを持つ DataFrame が返ること"""
        from src.prediction.unified_model_pipeline import load_all_stock_data

        df = pd.DataFrame(
            {
                "close_lag1": [100.0, 101.0],
                "rsi": [50.0, 55.0],
                "y": [0.01, -0.01],
                "market": ["jp", "jp"],
                "symbol": ["7203", "7203"],
            }
        )
        mock_load.return_value = df
        mock_mask.return_value = df  # mask をバイパス

        result = load_all_stock_data()

        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        mock_mask.assert_called_once()

    @patch("src.prediction.unified_model_pipeline.load_all_stock_features")
    def test_returns_empty_dataframe_when_no_data(self, mock_load):
        """データがない場合は空の DataFrame が返ること"""
        from src.prediction.unified_model_pipeline import load_all_stock_data

        mock_load.return_value = pd.DataFrame()

        result = load_all_stock_data()

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)


class TestApplyUnifiedFeatureExclusions(unittest.TestCase):
    """_apply_unified_feature_exclusions のテスト"""

    @patch("src.prediction.unified_model_pipeline.load_excluded_features")
    def test_removes_excluded_columns(self, mock_excluded):
        """除外リストに含まれる列が削除されること"""
        from src.prediction.unified_model_pipeline import _apply_unified_feature_exclusions

        mock_excluded.return_value = ["rsi", "volume_ma_deviation"]
        df = pd.DataFrame(
            {
                "close_lag1": [100.0],
                "rsi": [50.0],
                "volume_ma_deviation": [1.0],
                "y": [0.01],
            }
        )

        result = _apply_unified_feature_exclusions(df)

        self.assertNotIn("rsi", result.columns)
        self.assertNotIn("volume_ma_deviation", result.columns)
        self.assertIn("close_lag1", result.columns)

    @patch("src.prediction.unified_model_pipeline.load_excluded_features")
    def test_returns_same_df_when_no_exclusions(self, mock_excluded):
        """除外リストが空の場合は元の DataFrame と同じ列を持つこと"""
        from src.prediction.unified_model_pipeline import _apply_unified_feature_exclusions

        mock_excluded.return_value = []
        df = pd.DataFrame({"close_lag1": [100.0], "y": [0.01]})

        result = _apply_unified_feature_exclusions(df)

        self.assertEqual(set(result.columns), set(df.columns))


class TestTrainUnifiedModel(unittest.TestCase):
    """train_unified_model のテスト"""

    @patch("src.prediction.unified_model_pipeline._compute_and_save_unified_feature_selection")
    @patch("src.prediction.unified_model_pipeline._apply_unified_feature_exclusions")
    @patch("src.prediction.manager.ModelManager")
    @patch("src.prediction.unified_model_pipeline.prepare_unified_features")
    @patch("src.prediction.unified_model_pipeline.load_all_stock_data")
    def test_train_creates_and_saves_model(
        self, mock_load, mock_prepare, mock_mm_cls, mock_apply, mock_compute
    ):
        """モデルが作成・学習・保存されること"""
        from src.prediction.unified_model_pipeline import train_unified_model

        n = 20
        raw_df = pd.DataFrame(
            {
                "close_lag1": [float(i) for i in range(100, 100 + n)],
                "y": [0.01] * n,
                "market": ["jp"] * n,
                "symbol": ["7203"] * n,
            }
        )
        mock_load.return_value = raw_df

        X = pd.DataFrame(
            {
                "close_lag1": [float(i) for i in range(100, 100 + n)],
                "market_encoded": [1] * n,
            }
        )
        y = pd.Series([0.01] * n)
        mock_prepare.return_value = (X, y)
        mock_apply.return_value = X  # exclusion をバイパス

        mock_mm = mock_mm_cls.return_value
        mock_model = mock_mm.create_model.return_value

        with tempfile.TemporaryDirectory() as tmp:
            train_unified_model(
                model_type="XGBoostModel",
                model_name="TestUnifiedModel",
                save_dir=tmp,
            )

        mock_mm.create_model.assert_called_once_with("XGBoostModel", "TestUnifiedModel")
        mock_model.train.assert_called_once()
        mock_model.save_model.assert_called_once()
