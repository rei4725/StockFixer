"""ユニットテスト: 統合モデル用予測処理"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


def _make_feature_df(n: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    data = {
        "y": np.linspace(1000, 1100, n),
        "Close_lag1": np.linspace(998, 1098, n),
        "feature_a": np.random.rand(n),
        "feature_b": np.random.rand(n),
    }
    return pd.DataFrame(data, index=idx)


def _make_feature_df_with_atr(n: int = 30, atr_lag1_value: float = 5.0) -> pd.DataFrame:
    df = _make_feature_df(n)
    df["atr_lag1"] = atr_lag1_value
    return df


class TestLoadFeatureData(unittest.TestCase):
    def test_returns_dataframe_on_success(self):
        from src.prediction.predict_unified import load_feature_data

        df = _make_feature_df()
        with patch("src.prediction.predict_unified.load_stock_features", return_value=df):
            result = load_feature_data("jp", "7203")
        self.assertIsInstance(result, pd.DataFrame)

    def test_returns_none_on_exception(self):
        from src.prediction.predict_unified import load_feature_data

        with patch(
            "src.prediction.predict_unified.load_stock_features",
            side_effect=RuntimeError("db error"),
        ):
            result = load_feature_data("jp", "7203")
        self.assertIsNone(result)


class TestGetCachedModel(unittest.TestCase):
    def setUp(self):
        from src.prediction import predict_unified

        predict_unified._model_cache.clear()

    def test_loads_model_on_first_call(self):
        from src.prediction.predict_unified import get_cached_model

        mock_model = MagicMock()
        with patch(
            "src.prediction.unified_model_pipeline.load_unified_model", return_value=mock_model
        ) as mock_load:
            result = get_cached_model("TestModel")
        self.assertEqual(result, mock_model)
        mock_load.assert_called_once_with("TestModel")

    def test_caches_model_on_second_call(self):
        from src.prediction.predict_unified import get_cached_model

        mock_model = MagicMock()
        with patch(
            "src.prediction.unified_model_pipeline.load_unified_model", return_value=mock_model
        ) as mock_load:
            get_cached_model("TestModel2")
            get_cached_model("TestModel2")
        mock_load.assert_called_once()

    def test_returns_none_when_file_not_found(self):
        from src.prediction.predict_unified import get_cached_model

        with patch(
            "src.prediction.unified_model_pipeline.load_unified_model",
            side_effect=FileNotFoundError("not found"),
        ):
            result = get_cached_model("MissingModel2")
        self.assertIsNone(result)


class TestPreloadModels(unittest.TestCase):
    def setUp(self):
        from src.prediction import predict_unified

        predict_unified._model_cache.clear()

    def test_calls_get_cached_model_for_each_type(self):
        from src.prediction.predict_unified import preload_models

        mock_model = MagicMock()
        with patch(
            "src.prediction.predict_unified.get_cached_model", return_value=mock_model
        ) as mock_get:
            preload_models(["ModelA", "ModelB"])
        self.assertEqual(mock_get.call_count, 2)

    def test_uses_defaults_when_none(self):
        from src.prediction.predict_unified import preload_models

        with patch(
            "src.prediction.predict_unified.get_cached_model", return_value=None
        ) as mock_get:
            preload_models(None)
        self.assertEqual(mock_get.call_count, 2)


class TestPredictWithUnifiedModel(unittest.TestCase):
    def setUp(self):
        from src.prediction import predict_unified

        predict_unified._model_cache.clear()

    def test_returns_none_when_no_feature_data(self):
        from src.prediction.predict_unified import predict_with_unified_model

        with patch("src.prediction.predict_unified.load_feature_data", return_value=None):
            result = predict_with_unified_model("jp", "7203")
        self.assertIsNone(result)

    def test_returns_none_when_empty_feature_df(self):
        from src.prediction.predict_unified import predict_with_unified_model

        with patch("src.prediction.predict_unified.load_feature_data", return_value=pd.DataFrame()):
            result = predict_with_unified_model("jp", "7203")
        self.assertIsNone(result)

    def test_returns_none_when_no_y_column(self):
        from src.prediction.predict_unified import predict_with_unified_model

        df = pd.DataFrame({"feature_a": [1, 2, 3]})
        with patch("src.prediction.predict_unified.load_feature_data", return_value=df):
            result = predict_with_unified_model("jp", "7203")
        self.assertIsNone(result)

    def test_returns_none_when_no_models_available(self):
        from src.prediction.predict_unified import predict_with_unified_model

        df = _make_feature_df()
        with (
            patch("src.prediction.predict_unified.load_feature_data", return_value=df),
            patch("src.prediction.predict_unified.get_cached_model", return_value=None),
            patch(
                "src.prediction.predict_unified.get_ticker",
                return_value="7203.T",
            ),
        ):
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = pd.DataFrame(
                {"Close": [1050.0]}, index=pd.date_range("2024-01-31", periods=1)
            )
            with patch("src.prediction.predict_unified.yf.Ticker", return_value=mock_ticker):
                result = predict_with_unified_model("jp", "7203")
        self.assertIsNone(result)

    def test_returns_prediction_result_when_model_available(self):
        from src.prediction.predict_unified import predict_with_unified_model

        df = _make_feature_df()
        mock_model = MagicMock()
        mock_model.predict.return_value = pd.Series([0.02])

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [1050.0]}, index=pd.date_range("2024-01-31", periods=1)
        )

        with (
            patch("src.prediction.predict_unified.load_feature_data", return_value=df),
            patch("src.prediction.predict_unified.get_cached_model", return_value=mock_model),
            patch("src.prediction.predict_unified.get_ticker", return_value="7203.T"),
            patch("src.prediction.predict_unified.load_model_weights", return_value=[1.0]),
            patch("src.prediction.predict_unified.yf.Ticker", return_value=mock_ticker),
        ):
            result = predict_with_unified_model("jp", "7203")

        self.assertIsNotNone(result)
        self.assertEqual(result.market, "jp")
        self.assertEqual(result.symbol, "7203")

    def test_fallback_price_when_yf_fails(self):
        from src.prediction.predict_unified import predict_with_unified_model

        df = _make_feature_df()
        mock_model = MagicMock()
        mock_model.predict.return_value = pd.Series([0.01])

        with (
            patch("src.prediction.predict_unified.load_feature_data", return_value=df),
            patch("src.prediction.predict_unified.get_cached_model", return_value=mock_model),
            patch("src.prediction.predict_unified.get_ticker", return_value="7203.T"),
            patch("src.prediction.predict_unified.load_model_weights", return_value=[1.0]),
            patch(
                "src.prediction.predict_unified.yf.Ticker",
                side_effect=RuntimeError("network error"),
            ),
        ):
            result = predict_with_unified_model("jp", "7203")

        self.assertIsNotNone(result)

    def test_uses_horizon_specific_model_names(self):
        from src.prediction.predict_unified import predict_with_unified_model

        df = _make_feature_df()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.01]

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [1050.0]}, index=pd.date_range("2024-01-31", periods=1)
        )

        called_with = []

        def mock_get_cached(name):
            called_with.append(name)
            return mock_model

        with (
            patch("src.prediction.predict_unified.load_feature_data", return_value=df),
            patch("src.prediction.predict_unified.get_cached_model", side_effect=mock_get_cached),
            patch("src.prediction.predict_unified.get_ticker", return_value="7203.T"),
            patch("src.prediction.predict_unified.load_model_weights", return_value=[1.0, 1.0]),
            patch("src.prediction.predict_unified.yf.Ticker", return_value=mock_ticker),
        ):
            predict_with_unified_model("jp", "7203", horizon=5)

        self.assertTrue(any("5d" in name for name in called_with))


class TestAtrClipping(unittest.TestCase):
    def setUp(self):
        from src.prediction import predict_unified

        predict_unified._model_cache.clear()

    def test_outlier_prediction_is_clipped_using_atr_lag1(self):
        from src.prediction.predict_unified import predict_with_unified_model

        df = _make_feature_df_with_atr(atr_lag1_value=5.0)
        mock_model = MagicMock()
        mock_model.predict.return_value = pd.Series([0.50])  # 50%は外れ値

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [1000.0]}, index=pd.date_range("2024-01-31", periods=1)
        )

        with (
            patch("src.prediction.predict_unified.load_feature_data", return_value=df),
            patch("src.prediction.predict_unified.get_cached_model", return_value=mock_model),
            patch("src.prediction.predict_unified.get_ticker", return_value="7203.T"),
            patch("src.prediction.predict_unified.load_model_weights", return_value=[1.0]),
            patch("src.prediction.predict_unified.yf.Ticker", return_value=mock_ticker),
        ):
            result = predict_with_unified_model("jp", "7203")

        self.assertIsNotNone(result)
        # PREDICTION_ATR_CLIP_MULTIPLIER(既定3.0) * (5.0/1000.0) = 0.015
        self.assertAlmostEqual(result.diff_ratio, 0.015)
        self.assertAlmostEqual(result.avg_pred_price, 1015.0)

    def test_no_atr_lag1_column_does_not_clip(self):
        """atr_lag1列が無い場合は従来通りクリップされないこと"""
        from src.prediction.predict_unified import predict_with_unified_model

        df = _make_feature_df()  # atr_lag1 なし
        mock_model = MagicMock()
        mock_model.predict.return_value = pd.Series([0.50])

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [1000.0]}, index=pd.date_range("2024-01-31", periods=1)
        )

        with (
            patch("src.prediction.predict_unified.load_feature_data", return_value=df),
            patch("src.prediction.predict_unified.get_cached_model", return_value=mock_model),
            patch("src.prediction.predict_unified.get_ticker", return_value="7203.T"),
            patch("src.prediction.predict_unified.load_model_weights", return_value=[1.0]),
            patch("src.prediction.predict_unified.yf.Ticker", return_value=mock_ticker),
        ):
            result = predict_with_unified_model("jp", "7203")

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.diff_ratio, 0.50)


class TestPredictAllWithUnifiedModel(unittest.TestCase):
    def test_empty_symbols_returns_empty_df(self):
        from src.prediction.predict_unified import predict_all_with_unified_model

        with (
            patch("src.prediction.predict_unified.preload_models"),
            patch("src.prediction.predict_unified.get_all_symbols", return_value=[]),
        ):
            result = predict_all_with_unified_model()
        self.assertTrue(result.empty)

    def test_returns_dataframe_with_results(self):
        from src.prediction.predict_unified import predict_all_with_unified_model

        mock_row = pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "7203",
                    "current_price": 1000.0,
                    "avg_pred_price": 1020.0,
                    "diff_ratio": 0.02,
                    "model_count": 2,
                }
            ]
        )

        with (
            patch("src.prediction.predict_unified.preload_models"),
            patch(
                "src.prediction.predict_unified.get_all_symbols",
                return_value=[("jp", "7203")],
            ),
            patch(
                "src.prediction.predict_unified.predict_with_unified_model",
                return_value=mock_row,
            ),
        ):
            result = predict_all_with_unified_model()

        self.assertFalse(result.empty)


if __name__ == "__main__":
    unittest.main()
