"""model_training_pipeline モジュールのユニットテスト"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.prediction.training_pipeline import (
    _apply_feature_exclusions,
    _build_feature_selection_frame,
    _compute_training_metrics,
    _extract_mean_abs_shap_values,
    load_features_for_training,
    train_models_for_symbol,
    train_models_for_symbol_task,
)
from src.prediction.types import FeatureLoadResult, TrainingMetrics
from src.watchlist.types import BatchResult


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

    @patch("src.prediction.training_pipeline.load_stock_features")
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

    @patch("src.prediction.training_pipeline.load_stock_features")
    def test_skip_when_data_is_none(self, mock_load):
        """load_stock_features が None を返した場合は status=skip"""
        mock_load.return_value = None

        result = load_features_for_training("us", "AAPL")

        self.assertEqual(result.status, "skip")
        self.assertFalse(result.is_success)

    @patch("src.prediction.training_pipeline.load_stock_features")
    def test_skip_when_empty_df(self, mock_load):
        """空の DataFrame が返ってきた場合は status=skip"""
        mock_load.return_value = pd.DataFrame()

        result = load_features_for_training("us", "AAPL")

        self.assertEqual(result.status, "skip")

    @patch("src.prediction.training_pipeline.load_stock_features")
    def test_error_on_exception(self, mock_load):
        """DB例外時は status=error で error フィールドにメッセージが入ること"""
        mock_load.side_effect = RuntimeError("db connection error")

        result = load_features_for_training("us", "AAPL")

        self.assertEqual(result.status, "error")
        self.assertIsNotNone(result.error)
        self.assertIn("db connection error", result.error)

    @patch("src.prediction.training_pipeline.load_stock_features")
    def test_exclude_cols_not_in_X(self, mock_load):
        """y / market / symbol / date カラムは X から除外されること"""
        mock_load.return_value = self._make_stock_features_df()

        result = load_features_for_training("us", "AAPL")

        if result.is_success:
            for col in ("y", "market", "symbol", "date"):
                self.assertNotIn(col, result.X.columns)

    @patch("src.prediction.training_pipeline.load_stock_features")
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

    @patch("src.prediction.training_pipeline.get_earnings_dates")
    @patch("src.prediction.training_pipeline.load_stock_features")
    def test_horizon1_masks_earnings_window_rows(self, mock_load, mock_get_earnings):
        df = self._make_stock_features_df(periods=20)
        df.loc[:, "date"] = df.index
        mock_load.return_value = df
        mock_get_earnings.return_value = pd.DatetimeIndex([pd.Timestamp("2024-01-10")])

        result = load_features_for_training("us", "AAPL", horizon=1)

        self.assertTrue(result.is_success)
        self.assertLess(len(result.X), len(df))
        self.assertNotIn(pd.Timestamp("2024-01-10"), result.X.index)

    @patch("src.prediction.training_pipeline.load_excluded_features")
    @patch("src.prediction.training_pipeline.load_stock_features")
    def test_load_features_applies_saved_exclusions(self, mock_load, mock_excluded):
        mock_load.return_value = self._make_stock_features_df()
        mock_excluded.return_value = ["volume"]

        result = load_features_for_training("us", "AAPL", horizon=1)

        self.assertTrue(result.is_success)
        self.assertNotIn("volume", result.X.columns)


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


class TestFeatureSelectionHelpers(unittest.TestCase):
    def test_apply_feature_exclusions_drops_matching_columns(self):
        X = pd.DataFrame({"f1": [1, 2], "f2": [3, 4], "f3": [5, 6]})

        with patch("src.prediction.training_pipeline.load_excluded_features", return_value=["f2"]):
            filtered = _apply_feature_exclusions(X, "us", "AAPL")

        self.assertEqual(filtered.columns.tolist(), ["f1", "f3"])

    def test_build_feature_selection_frame_preserves_protected_feature(self):
        X = pd.DataFrame({f"f{i}": [i, i + 1] for i in range(12)})

        selection_df = _build_feature_selection_frame(
            X,
            importance_mean=np.array([1.0] + [0.5] * 10 + [-1.0]),
            importance_std=np.zeros(12),
            protected_features={"f11"},
        )

        row = selection_df.loc[selection_df["feature"] == "f11"].iloc[0]
        self.assertTrue(row["protected_by_shap"])
        self.assertFalse(row["is_excluded"])


class TestTrainModelsForSymbol(unittest.TestCase):
    """train_models_for_symbol 関数のテスト"""

    def _make_feature_result(self, market="us", symbol="AAPL", n=100):
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        X = pd.DataFrame({"f1": range(n), "f2": np.linspace(0, 1, n)}, index=dates)
        y = pd.Series(np.random.randn(n) * 0.01, index=dates)
        return FeatureLoadResult(status="success", market=market, symbol=symbol, X=X, y=y)

    @patch("src.prediction.training_pipeline.save_model_metrics")
    @patch("src.prediction.training_pipeline.ModelManager")
    @patch("src.prediction.training_pipeline.load_features_for_training")
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

    @patch("src.prediction.training_pipeline.load_features_for_training")
    def test_skip_propagated_from_load(self, mock_load):
        """load_features_for_training が skip を返した場合、skip が伝搬すること"""
        mock_load.return_value = FeatureLoadResult(
            status="skip", market="us", symbol="AAPL", reason="データなし"
        )

        result = train_models_for_symbol("us", "AAPL")

        self.assertEqual(result["status"], "skip")
        self.assertEqual(result["market"], "us")

    @patch("src.prediction.training_pipeline.load_features_for_training")
    def test_exception_returns_error(self, mock_load):
        """予期しない例外発生時は status=error が返ること"""
        mock_load.side_effect = RuntimeError("unexpected failure")

        result = train_models_for_symbol("us", "AAPL")

        self.assertEqual(result["status"], "error")
        self.assertIn("error", result)

    @patch("src.prediction.training_pipeline.save_model_metrics")
    @patch("src.prediction.training_pipeline.ModelManager")
    @patch("src.prediction.training_pipeline.load_features_for_training")
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

    @patch("src.prediction.training_pipeline._compute_and_save_shap")
    @patch("src.prediction.training_pipeline.save_model_metrics")
    @patch("src.prediction.training_pipeline.ModelManager")
    @patch("src.prediction.training_pipeline.load_features_for_training")
    def test_shap_notification_sent_for_each_trained_model(
        self,
        mock_load,
        mock_mm_cls,
        mock_save_metrics,
        mock_compute_shap,
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

        result = train_models_for_symbol("us", "AAPL")

        # SHAP計算はモデルごと（XGBoost + LightGBM）に2回実行される
        self.assertEqual(mock_compute_shap.call_count, 2)
        # 個別Discord通知ではなく、shap_resultsとして戻り値に集約される
        self.assertIn("shap_results", result)
        self.assertEqual(len(result["shap_results"]), 2)
        for entry in result["shap_results"]:
            self.assertEqual(entry["market"], "us")
            self.assertEqual(entry["symbol"], "AAPL")
            self.assertIn("model_name", entry)
            self.assertFalse(entry["shap_top_bottom"].empty)


class TestTrainModelsForSymbolTask(unittest.TestCase):
    """train_models_for_symbol_task 関数のテスト"""

    @patch("src.prediction.training_pipeline.train_models_for_symbol")
    def test_accepts_symbol_task(self, mock_train):
        """SymbolTask を渡せること"""
        from src.watchlist.types import SymbolTask

        mock_train.return_value = {"status": "success", "market": "us", "symbol": "AAPL"}
        task = SymbolTask(market="us", symbol="AAPL", horizon=1)

        result = train_models_for_symbol_task(task)

        mock_train.assert_called_once_with("us", "AAPL", 1, shadow_mode=False, use_transformer=False)
        self.assertEqual(result["status"], "success")

    @patch("src.prediction.training_pipeline.train_models_for_symbol")
    def test_accepts_dict_task(self, mock_train):
        """dict 形式のタスクを渡せること"""
        mock_train.return_value = {"status": "success", "market": "jp", "symbol": "7203"}
        task = {"market": "jp", "symbol": "7203", "horizon": 3}

        train_models_for_symbol_task(task)

        mock_train.assert_called_once_with("jp", "7203", 3, shadow_mode=False, use_transformer=False)

    @patch("src.prediction.training_pipeline.train_models_for_symbol")
    def test_dict_task_default_horizon(self, mock_train):
        """dict に horizon キーがない場合は horizon=1 がデフォルトであること"""
        mock_train.return_value = {"status": "success", "market": "us", "symbol": "MSFT"}

        train_models_for_symbol_task({"market": "us", "symbol": "MSFT"})

        mock_train.assert_called_once_with("us", "MSFT", 1, shadow_mode=False, use_transformer=False)


class TestComputeAndSaveShap(unittest.TestCase):
    """_compute_and_save_shap のテスト"""

    @patch("src.prediction.training_pipeline.save_shap_values")
    def test_saves_shap_values_to_db(self, mock_save):
        """SHAP 値が DB に保存されること"""
        import sys

        from src.prediction.training_pipeline import _compute_and_save_shap

        mock_shap = MagicMock()
        mock_explainer_inst = MagicMock()
        mock_shap.TreeExplainer.return_value = mock_explainer_inst
        mock_explainer_inst.shap_values.return_value = np.random.randn(10, 3)

        mock_model = MagicMock()
        X = pd.DataFrame(
            {
                "close_lag1": np.random.randn(10),
                "rsi": np.random.randn(10),
                "sma_ratio": np.random.randn(10),
            }
        )

        with patch.dict(sys.modules, {"shap": mock_shap}):
            _compute_and_save_shap(mock_model, X, "jp", "7203", "XGBoostModel", "2026-04-18")

        mock_save.assert_called_once()

    def test_no_exception_on_shap_error(self):
        """SHAP 計算エラー時も例外が外に伝播しないこと"""
        import sys

        from src.prediction.training_pipeline import _compute_and_save_shap

        mock_shap = MagicMock()
        mock_shap.TreeExplainer.side_effect = Exception("SHAP計算エラー")

        mock_model = MagicMock()
        X = pd.DataFrame({"close_lag1": np.random.randn(10)})

        # 例外が発生しないこと（内部で logger.warning してから空 DataFrame を返す設計）
        with patch.dict(sys.modules, {"shap": mock_shap}):
            result = _compute_and_save_shap(
                mock_model, X, "jp", "7203", "XGBoostModel", "2026-04-18"
            )

        self.assertIsInstance(result, pd.DataFrame)

    @patch("src.prediction.training_pipeline.save_shap_values")
    def test_returns_top_and_bottom_features(self, mock_save):
        """上位・下位 N 件の特徴量 DataFrame が返ること"""
        import sys

        from src.prediction.training_pipeline import _compute_and_save_shap

        n_features = 8
        mock_shap = MagicMock()
        mock_explainer_inst = MagicMock()
        mock_shap.TreeExplainer.return_value = mock_explainer_inst
        mock_explainer_inst.shap_values.return_value = np.random.randn(20, n_features)

        mock_model = MagicMock()
        X = pd.DataFrame({f"feat_{i}": np.random.randn(20) for i in range(n_features)})

        with patch.dict(sys.modules, {"shap": mock_shap}):
            result = _compute_and_save_shap(
                mock_model, X, "jp", "7203", "XGBoostModel", "2026-04-18", top_n=3
            )

        # top_n=3 なので上位3 + 下位3 = 最大6件
        self.assertIsInstance(result, pd.DataFrame)
        self.assertLessEqual(len(result), 6)


class TestLoadFeaturesForTrainingHorizon(unittest.TestCase):
    """load_features_for_training の horizon > 1 テスト"""

    @patch("src.prediction.training_pipeline.get_earnings_dates")
    @patch("src.prediction.training_pipeline.add_earnings_flag")
    @patch("src.market_data.technical.create_basic_lag_features")
    @patch("src.market_data.technical.add_technical_indicators")
    @patch("src.utils.db.load_raw_ohlcv")
    def test_horizon_3_uses_raw_ohlcv(
        self, mock_raw, mock_ti, mock_lag, mock_add_earnings, mock_earnings_dates
    ):
        """horizon=3 では market_data_raw からデータを読み込むこと"""
        from src.prediction.training_pipeline import load_features_for_training

        n = 50
        df = pd.DataFrame(
            {
                "Open": np.random.rand(n) * 100 + 100,
                "High": np.random.rand(n) * 100 + 110,
                "Low": np.random.rand(n) * 100 + 90,
                "Close": np.random.rand(n) * 100 + 100,
                "Volume": np.random.randint(1000, 9999, n).astype(float),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="B"),
        )

        mock_raw.return_value = df
        mock_add_earnings.return_value = df
        mock_ti.return_value = df
        mock_earnings_dates.return_value = pd.DatetimeIndex([])

        X = pd.DataFrame(
            {"close_lag1": np.random.randn(n - 4)},
            index=pd.date_range("2024-01-01", periods=n - 4, freq="B"),
        )
        y = pd.Series(
            np.random.randn(n - 4),
            index=pd.date_range("2024-01-01", periods=n - 4, freq="B"),
        )
        mock_lag.return_value = (X, y)

        load_features_for_training("jp", "7203", horizon=3)
        mock_raw.assert_called_once_with("jp", "7203")

    @patch("src.prediction.training_pipeline.get_earnings_dates")
    @patch("src.prediction.training_pipeline.add_earnings_flag")
    @patch("src.market_data.technical.create_basic_lag_features")
    @patch("src.market_data.technical.add_technical_indicators")
    @patch("src.utils.db.load_raw_ohlcv")
    def test_horizon_3_returns_success(
        self, mock_raw, mock_ti, mock_lag, mock_add_earnings, mock_earnings_dates
    ):
        """horizon=3 で正常データがあれば status=success が返ること"""
        from src.prediction.training_pipeline import load_features_for_training

        n = 50
        df = pd.DataFrame(
            {
                "Open": np.random.rand(n) + 100,
                "High": np.random.rand(n) + 110,
                "Low": np.random.rand(n) + 90,
                "Close": np.random.rand(n) + 100,
                "Volume": np.random.randint(1000, 9999, n).astype(float),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="B"),
        )

        mock_raw.return_value = df
        mock_add_earnings.return_value = df
        mock_ti.return_value = df
        mock_earnings_dates.return_value = pd.DatetimeIndex([])

        X = pd.DataFrame(
            {"close_lag1": np.random.randn(n - 4), "rsi": np.random.randn(n - 4)},
            index=pd.date_range("2024-01-01", periods=n - 4, freq="B"),
        )
        y = pd.Series(
            np.random.randn(n - 4),
            index=pd.date_range("2024-01-01", periods=n - 4, freq="B"),
        )
        mock_lag.return_value = (X, y)

        result = load_features_for_training("jp", "7203", horizon=3)
        self.assertEqual(result.status, "success")
        self.assertTrue(result.is_success)

    @patch("src.utils.db.load_raw_ohlcv")
    def test_horizon_3_skip_when_ohlcv_none(self, mock_raw):
        """horizon=3 で OHLCV が None の場合は status=skip が返ること"""
        from src.prediction.training_pipeline import load_features_for_training

        mock_raw.return_value = None

        result = load_features_for_training("jp", "7203", horizon=3)
        self.assertEqual(result.status, "skip")
        self.assertFalse(result.is_success)


# ──────────────────────────────────────────────────────────────────
# run_model_batch
# ──────────────────────────────────────────────────────────────────


class TestRunModelBatch(unittest.TestCase):
    """run_model_batch のテスト"""

    @patch("src.watchlist.batch_runner.run_parallel")
    @patch("src.watchlist.batch_runner.load_target_symbols")
    def test_returns_early_when_no_symbols(self, mock_symbols, mock_parallel):
        """銘柄がない場合、早期リターンすること（run_parallel 未呼び出し）"""
        from src.prediction.training_pipeline import run_model_batch

        mock_symbols.return_value = []
        run_model_batch(horizon=1)  # 例外なし
        mock_parallel.assert_not_called()

    @patch("src.watchlist.batch_runner.print_summary")
    @patch("src.watchlist.batch_runner.run_parallel")
    @patch("src.watchlist.batch_runner.load_target_symbols")
    def test_runs_parallel_when_symbols_available(self, mock_symbols, mock_parallel, mock_print):
        """銘柄がある場合、run_parallel が呼ばれること"""
        from src.prediction.training_pipeline import run_model_batch
        from src.watchlist.types import SymbolTask

        mock_symbols.return_value = [SymbolTask(market="jp", symbol="7203")]
        mock_parallel.return_value = BatchResult(succeeded=[], failed=[], skipped=[])
        run_model_batch(horizon=1)
        mock_parallel.assert_called_once()

    @patch("src.watchlist.batch_runner.print_summary")
    @patch("src.prediction.training_pipeline.save_model_metrics")
    @patch("src.prediction.training_pipeline.ModelManager")
    @patch("src.watchlist.batch_runner.run_parallel")
    @patch("src.watchlist.batch_runner.load_target_symbols")
    def test_trains_models_for_successful_loads(
        self, mock_symbols, mock_parallel, mock_mm_cls, mock_save, mock_print
    ):
        """成功した特徴量ロードに対してモデルが学習されること"""
        from src.prediction.training_pipeline import run_model_batch
        from src.prediction.types import FeatureLoadResult
        from src.watchlist.types import SymbolTask

        mock_symbols.return_value = [SymbolTask(market="jp", symbol="7203")]

        X = pd.DataFrame({"f1": np.random.randn(50), "f2": np.random.randn(50)})
        y = pd.Series(np.random.randn(50) * 0.01)
        load_result = FeatureLoadResult(status="success", market="jp", symbol="7203", X=X, y=y)
        mock_parallel.return_value = BatchResult(succeeded=[load_result], failed=[], skipped=[])

        mock_model = MagicMock()
        mock_model.predict.return_value = pd.Series(np.zeros(50))
        mock_mm = MagicMock()
        mock_mm_cls.return_value = mock_mm
        mock_mm.get_model.return_value = mock_model

        run_model_batch(horizon=1)

        self.assertGreaterEqual(mock_mm.create_model.call_count, 2)
        self.assertGreaterEqual(mock_mm.train_model.call_count, 2)

    @patch("src.watchlist.batch_runner.print_summary")
    @patch("src.watchlist.batch_runner.run_parallel")
    @patch("src.watchlist.batch_runner.load_target_symbols")
    def test_horizon_attached_to_tasks(self, mock_symbols, mock_parallel, mock_print):
        """horizon パラメータが SymbolTask に付与されること"""
        from src.prediction.training_pipeline import run_model_batch
        from src.watchlist.types import SymbolTask

        mock_symbols.return_value = [SymbolTask(market="jp", symbol="7203")]
        mock_parallel.return_value = BatchResult(succeeded=[], failed=[], skipped=[])

        run_model_batch(horizon=5)

        mock_parallel.assert_called_once()
        call_kwargs = mock_parallel.call_args.kwargs
        tasks = call_kwargs.get("tasks", [])
        self.assertGreater(len(tasks), 0)
        first_task = tasks[0]
        horizon_val = (
            first_task.horizon if hasattr(first_task, "horizon") else first_task.get("horizon")
        )
        self.assertEqual(horizon_val, 5)


# ──────────────────────────────────────────────────────────────────
# train_all_models
# ──────────────────────────────────────────────────────────────────


class TestTrainAllModels(unittest.TestCase):
    """train_all_models のテスト"""

    @patch("src.prediction.training_pipeline._train_models_for_horizon")
    def test_calls_train_for_each_horizon(self, mock_train):
        """各 horizon に対して _train_models_for_horizon が呼ばれること"""
        from src.prediction.training_pipeline import train_all_models

        train_all_models(horizons=[1, 7])
        self.assertEqual(mock_train.call_count, 2)
        calls = [c[0][0] for c in mock_train.call_args_list]
        self.assertIn(1, calls)
        self.assertIn(7, calls)

    @patch("src.prediction.training_pipeline._train_models_for_horizon")
    def test_exception_per_horizon_does_not_stop_others(self, mock_train):
        """1つの horizon で例外が出ても他の horizon が実行されること"""
        from src.prediction.training_pipeline import train_all_models

        mock_train.side_effect = [Exception("学習エラー"), None]
        train_all_models(horizons=[1, 7])
        self.assertEqual(mock_train.call_count, 2)

    @patch("src.prediction.training_pipeline._train_models_for_horizon")
    def test_default_horizon_is_1(self, mock_train):
        """引数なしでは horizon=1 が使われること"""
        from src.prediction.training_pipeline import train_all_models

        train_all_models()
        mock_train.assert_called_once_with(1, max_workers=3)


# ──────────────────────────────────────────────────────────────────
# _train_models_for_horizon
# ──────────────────────────────────────────────────────────────────


class TestTrainModelsForHorizon(unittest.TestCase):
    """_train_models_for_horizon のテスト"""

    @patch("src.prediction.training_pipeline.train_models_for_symbol")
    @patch("src.prediction.training_pipeline.load_features_for_training")
    @patch("src.prediction.training_pipeline.load_target_symbols")
    def test_trains_when_features_available(self, mock_symbols, mock_load, mock_train):
        """特徴量が存在する場合にモデルが学習されること"""
        from src.prediction.training_pipeline import _train_models_for_horizon
        from src.prediction.types import FeatureLoadResult
        from src.watchlist.types import SymbolTask

        mock_symbols.return_value = [SymbolTask(market="jp", symbol="7203")]
        X = pd.DataFrame({"close_lag1": np.random.randn(100)})
        y = pd.Series(np.random.randn(100))
        mock_load.return_value = FeatureLoadResult(
            market="jp", symbol="7203", status="success", X=X, y=y
        )
        mock_train.return_value = {"market": "jp", "symbol": "7203", "status": "success"}

        _train_models_for_horizon(horizon=1, max_workers=1)
        mock_train.assert_called_once_with("jp", "7203", horizon=1)

    @patch("src.prediction.training_pipeline.train_models_for_symbol")
    @patch("src.prediction.training_pipeline.load_features_for_training")
    @patch("src.prediction.training_pipeline.load_target_symbols")
    def test_skips_when_features_unavailable(self, mock_symbols, mock_load, mock_train):
        """特徴量がない場合はスキップされること（例外なし）"""
        from src.prediction.training_pipeline import _train_models_for_horizon
        from src.prediction.types import FeatureLoadResult
        from src.watchlist.types import SymbolTask

        mock_symbols.return_value = [SymbolTask(market="jp", symbol="7203")]
        mock_load.return_value = FeatureLoadResult(
            market="jp", symbol="7203", status="error", X=pd.DataFrame(), y=pd.Series(dtype=float)
        )

        _train_models_for_horizon(horizon=1, max_workers=1)  # 例外なし
        mock_train.assert_not_called()

    @patch("src.prediction.training_pipeline.load_target_symbols")
    def test_returns_empty_when_no_symbols(self, mock_symbols):
        """銘柄がない場合、空リストが返ること"""
        from src.prediction.training_pipeline import _train_models_for_horizon

        mock_symbols.return_value = []
        results = _train_models_for_horizon(horizon=1)
        self.assertEqual(results, [])

    @patch("src.prediction.training_pipeline.train_models_for_symbol")
    @patch("src.prediction.training_pipeline.load_features_for_training")
    @patch("src.prediction.training_pipeline.load_target_symbols")
    def test_exception_in_one_symbol_continues_others(self, mock_symbols, mock_load, mock_train):
        """1銘柄でエラーが出ても残りが処理されること"""
        from src.prediction.training_pipeline import _train_models_for_horizon
        from src.prediction.types import FeatureLoadResult
        from src.watchlist.types import SymbolTask

        mock_symbols.return_value = [
            SymbolTask(market="jp", symbol="7203"),
            SymbolTask(market="jp", symbol="9984"),
        ]
        X = pd.DataFrame({"f1": np.random.randn(50)})
        y = pd.Series(np.random.randn(50))
        mock_load.return_value = FeatureLoadResult(
            market="jp", symbol="7203", status="success", X=X, y=y
        )
        mock_train.side_effect = [Exception("学習失敗"), {"status": "success"}]

        results = _train_models_for_horizon(horizon=1, max_workers=1)
        self.assertEqual(len(results), 2)
        statuses = [r.get("status") for r in results]
        self.assertIn("error", statuses)


if __name__ == "__main__":
    unittest.main()
