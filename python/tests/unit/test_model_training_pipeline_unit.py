"""model_training_pipeline モジュールのユニットテスト"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.domain.types import FeatureLoadResult, TrainingMetrics
from src.services.model_training_pipeline import (
    _compute_training_metrics,
    _extract_mean_abs_shap_values,
    load_features_for_training,
    train_models_for_symbol,
    train_models_for_symbol_task,
)


class TestComputeTrainingMetrics(unittest.TestCase):
    """_compute_training_metrics のテスト"""

    def test_perfect_prediction_rmse_zero(self):
        """完全一致する場合 RMSE=0 になること"""
        y = pd.Series([0.01, -0.02, 0.03, -0.01])
        metrics = _compute_training_metrics(y, y)
        self.assertAlmostEqual(metrics.rmse, 0.0)

    def test_perfect_prediction_accuracy_one(self):
        """完全一致する場合 directional_accuracy=1.0 になること"""
        y = pd.Series([0.01, -0.02, 0.03, -0.01])
        metrics = _compute_training_metrics(y, y)
        self.assertAlmostEqual(metrics.directional_accuracy, 1.0)

    def test_n_samples_matches_series_length(self):
        """n_samples が Series の長さと一致すること"""
        y = pd.Series(np.random.randn(200) * 0.01)
        metrics = _compute_training_metrics(y, y)
        self.assertEqual(metrics.n_samples, 200)

    def test_directional_accuracy_half(self):
        """2件中2件方向不一致の場合に正解率 0.0 になること"""
        y_true = pd.Series([1.0, 1.0])
        y_pred = pd.Series([-1.0, -1.0])
        metrics = _compute_training_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics.directional_accuracy, 0.0)

    def test_returns_training_metrics_type(self):
        """戻り値が TrainingMetrics 型であること"""
        y = pd.Series([0.01, -0.02])
        m = _compute_training_metrics(y, y)
        self.assertIsInstance(m, TrainingMetrics)

    def test_rmse_positive(self):
        """RMSE は常に 0 以上であること"""
        y_true = pd.Series([0.01, -0.02, 0.03])
        y_pred = pd.Series([0.00, -0.01, 0.05])
        metrics = _compute_training_metrics(y_true, y_pred)
        self.assertGreaterEqual(metrics.rmse, 0.0)


class TestLoadFeaturesForTraining(unittest.TestCase):
    """load_features_for_training 関数のテスト"""

    def _make_stock_features_df(self, periods=100):
        dates = pd.date_range("2024-01-01", periods=periods, freq="D")
        return pd.DataFrame(
            {
                "y": np.random.randn(periods) * 0.01,
                "close": np.linspace(100, 120, periods),
                "volume": np.random.randint(1000, 5000, periods),
                "market": ["us"] * periods,
                "symbol": ["AAPL"] * periods,
                "date": dates,
            },
            index=dates,
        )

    @patch("src.services.model_training_pipeline.load_stock_features")
    def test_success_with_valid_data(self, mock_load):
        """DBにデータがある場合は status=success で X/y が付与されること"""
        mock_load.return_value = self._make_stock_features_df()

        result = load_features_for_training("us", "AAPL", horizon=1)

        self.assertEqual(result.status, "success")
        self.assertTrue(result.is_success)
        self.assertIsNotNone(result.X)
        self.assertIsNotNone(result.y)
        self.assertEqual(result.market, "us")
        self.assertEqual(result.symbol, "AAPL")

    @patch("src.services.model_training_pipeline.load_stock_features")
    def test_skip_when_data_is_none(self, mock_load):
        """load_stock_features が None を返した場合は status=skip"""
        mock_load.return_value = None

        result = load_features_for_training("us", "AAPL")

        self.assertEqual(result.status, "skip")
        self.assertFalse(result.is_success)

    @patch("src.services.model_training_pipeline.load_stock_features")
    def test_skip_when_empty_df(self, mock_load):
        """空の DataFrame が返ってきた場合は status=skip"""
        mock_load.return_value = pd.DataFrame()

        result = load_features_for_training("us", "AAPL")

        self.assertEqual(result.status, "skip")

    @patch("src.services.model_training_pipeline.load_stock_features")
    def test_error_on_exception(self, mock_load):
        """DB例外時は status=error で error フィールドにメッセージが入ること"""
        mock_load.side_effect = RuntimeError("db connection error")

        result = load_features_for_training("us", "AAPL")

        self.assertEqual(result.status, "error")
        self.assertIsNotNone(result.error)
        self.assertIn("db connection error", result.error)

    @patch("src.services.model_training_pipeline.load_stock_features")
    def test_exclude_cols_not_in_X(self, mock_load):
        """y / market / symbol / date カラムは X から除外されること"""
        mock_load.return_value = self._make_stock_features_df()

        result = load_features_for_training("us", "AAPL")

        if result.is_success:
            for col in ("y", "market", "symbol", "date"):
                self.assertNotIn(col, result.X.columns)

    @patch("src.services.model_training_pipeline.load_stock_features")
    def test_feature_column_names_normalized(self, mock_load):
        """特徴量カラム名に含まれる非英数字が '_' に置換されること"""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        df = pd.DataFrame(
            {
                "y": np.random.randn(50) * 0.01,
                "close-price": np.linspace(100, 110, 50),
                "vol/day": np.random.randint(1000, 5000, 50),
                "market": "us",
                "symbol": "AAPL",
                "date": dates,
            },
            index=dates,
        )
        mock_load.return_value = df

        result = load_features_for_training("us", "AAPL")

        if result.is_success:
            for col in result.X.columns:
                self.assertNotIn("-", col)
                self.assertNotIn("/", col)

    @patch("src.services.model_training_pipeline.get_earnings_dates")
    @patch("src.services.model_training_pipeline.load_stock_features")
    def test_horizon1_masks_earnings_window_rows(self, mock_load, mock_get_earnings):
        df = self._make_stock_features_df(periods=20)
        df.loc[:, "date"] = df.index
        mock_load.return_value = df
        mock_get_earnings.return_value = pd.DatetimeIndex([pd.Timestamp("2024-01-10")])

        result = load_features_for_training("us", "AAPL", horizon=1)

        self.assertTrue(result.is_success)
        self.assertLess(len(result.X), len(df))
        self.assertNotIn(pd.Timestamp("2024-01-10"), result.X.index)


class _DummyExplanation:
    def __init__(self, values):
        self.values = values


class TestExtractMeanAbsShapValues(unittest.TestCase):
    """_extract_mean_abs_shap_values のテスト"""

    def test_accepts_2d_numpy_array(self):
        shap_values = np.array([[1.0, -2.0], [-3.0, 4.0]])

        result = _extract_mean_abs_shap_values(shap_values, feature_count=2)

        np.testing.assert_allclose(result, np.array([2.0, 3.0]))

    def test_accepts_explanation_like_object(self):
        shap_values = _DummyExplanation(np.array([[1.0, -3.0], [-1.0, 5.0]]))

        result = _extract_mean_abs_shap_values(shap_values, feature_count=2)

        np.testing.assert_allclose(result, np.array([1.0, 4.0]))

    def test_accepts_list_outputs_and_averages_across_outputs(self):
        shap_values = [
            np.array([[1.0, -3.0], [-1.0, 5.0]]),
            np.array([[2.0, -1.0], [-2.0, 7.0]]),
        ]

        result = _extract_mean_abs_shap_values(shap_values, feature_count=2)

        np.testing.assert_allclose(result, np.array([1.5, 4.0]))

    def test_raises_when_feature_axis_cannot_be_found(self):
        with self.assertRaises(ValueError):
            _extract_mean_abs_shap_values(np.array([[1.0, 2.0, 3.0]]), feature_count=2)


class TestTrainModelsForSymbol(unittest.TestCase):
    """train_models_for_symbol 関数のテスト"""

    def _make_feature_result(self, market="us", symbol="AAPL", n=100):
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        X = pd.DataFrame({"f1": range(n), "f2": np.linspace(0, 1, n)}, index=dates)
        y = pd.Series(np.random.randn(n) * 0.01, index=dates)
        return FeatureLoadResult(status="success", market=market, symbol=symbol, X=X, y=y)

    @patch("src.services.model_training_pipeline.save_model_metrics")
    @patch("src.services.model_training_pipeline.ModelManager")
    @patch("src.services.model_training_pipeline.load_features_for_training")
    def test_success_returns_success_dict(self, mock_load, mock_mm_cls, mock_save_metrics):
        """正常完了時は status=success の dict が返ること"""
        mock_load.return_value = self._make_feature_result()
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        mock_model = MagicMock()
        mock_model.predict.return_value = pd.Series(np.zeros(100), index=dates)
        mock_mm = MagicMock()
        mock_mm.get_model.return_value = mock_model
        mock_mm_cls.return_value = mock_mm

        result = train_models_for_symbol("us", "AAPL")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["market"], "us")
        self.assertEqual(result["symbol"], "AAPL")

    @patch("src.services.model_training_pipeline.load_features_for_training")
    def test_skip_propagated_from_load(self, mock_load):
        """load_features_for_training が skip を返した場合、skip が伝搬すること"""
        mock_load.return_value = FeatureLoadResult(
            status="skip", market="us", symbol="AAPL", reason="データなし"
        )

        result = train_models_for_symbol("us", "AAPL")

        self.assertEqual(result["status"], "skip")
        self.assertEqual(result["market"], "us")

    @patch("src.services.model_training_pipeline.load_features_for_training")
    def test_exception_returns_error(self, mock_load):
        """予期しない例外発生時は status=error が返ること"""
        mock_load.side_effect = RuntimeError("unexpected failure")

        result = train_models_for_symbol("us", "AAPL")

        self.assertEqual(result["status"], "error")
        self.assertIn("error", result)

    @patch("src.services.model_training_pipeline.save_model_metrics")
    @patch("src.services.model_training_pipeline.ModelManager")
    @patch("src.services.model_training_pipeline.load_features_for_training")
    def test_model_manager_create_called_twice(self, mock_load, mock_mm_cls, mock_save_metrics):
        """XGBoost / LightGBM の 2 モデルが作成されること"""
        mock_load.return_value = self._make_feature_result()
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        mock_model = MagicMock()
        mock_model.predict.return_value = pd.Series(np.zeros(100), index=dates)
        mock_mm = MagicMock()
        mock_mm.get_model.return_value = mock_model
        mock_mm_cls.return_value = mock_mm

        train_models_for_symbol("us", "AAPL")

        self.assertEqual(mock_mm.create_model.call_count, 2)
        self.assertEqual(mock_mm.train_model.call_count, 2)

    @patch("src.api.discord_utils.send_shap_notification")
    @patch("src.services.model_training_pipeline._compute_and_save_shap")
    @patch("src.services.model_training_pipeline.save_model_metrics")
    @patch("src.services.model_training_pipeline.ModelManager")
    @patch("src.services.model_training_pipeline.load_features_for_training")
    def test_shap_notification_sent_for_each_trained_model(
        self,
        mock_load,
        mock_mm_cls,
        mock_save_metrics,
        mock_compute_shap,
        mock_send_shap,
    ):
        mock_load.return_value = self._make_feature_result()
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        mock_model = MagicMock()
        mock_model.predict.return_value = pd.Series(np.zeros(100), index=dates)
        mock_mm = MagicMock()
        mock_mm.get_model.return_value = mock_model
        mock_mm_cls.return_value = mock_mm
        mock_compute_shap.return_value = pd.DataFrame(
            {
                "feature": ["f1", "f2"],
                "shap_mean": [0.5, 0.1],
                "shap_rank": [1, 2],
            }
        )

        train_models_for_symbol("us", "AAPL")

        self.assertEqual(mock_compute_shap.call_count, 2)
        self.assertEqual(mock_send_shap.call_count, 2)


class TestTrainModelsForSymbolTask(unittest.TestCase):
    """train_models_for_symbol_task 関数のテスト"""

    @patch("src.services.model_training_pipeline.train_models_for_symbol")
    def test_accepts_symbol_task(self, mock_train):
        """SymbolTask を渡せること"""
        from src.domain.types import SymbolTask

        mock_train.return_value = {"status": "success", "market": "us", "symbol": "AAPL"}
        task = SymbolTask(market="us", symbol="AAPL", horizon=1)

        result = train_models_for_symbol_task(task)

        mock_train.assert_called_once_with("us", "AAPL", 1)
        self.assertEqual(result["status"], "success")

    @patch("src.services.model_training_pipeline.train_models_for_symbol")
    def test_accepts_dict_task(self, mock_train):
        """dict 形式のタスクを渡せること"""
        mock_train.return_value = {"status": "success", "market": "jp", "symbol": "7203"}
        task = {"market": "jp", "symbol": "7203", "horizon": 3}

        train_models_for_symbol_task(task)

        mock_train.assert_called_once_with("jp", "7203", 3)

    @patch("src.services.model_training_pipeline.train_models_for_symbol")
    def test_dict_task_default_horizon(self, mock_train):
        """dict に horizon キーがない場合は horizon=1 がデフォルトであること"""
        mock_train.return_value = {"status": "success", "market": "us", "symbol": "MSFT"}

        train_models_for_symbol_task({"market": "us", "symbol": "MSFT"})

        mock_train.assert_called_once_with("us", "MSFT", 1)


if __name__ == "__main__":
    unittest.main()
