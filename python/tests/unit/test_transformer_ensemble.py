"""TransformerModel のアンサンブル統合テスト (Issue #257)"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.prediction.types import FeatureLoadResult


def _make_feature_result(market="us", symbol="AAPL", n=100):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    X = pd.DataFrame({"f1": range(n), "f2": np.linspace(0, 1, n)}, index=dates)
    y = pd.Series(np.random.randn(n) * 0.01, index=dates)
    return FeatureLoadResult(status="success", market=market, symbol=symbol, X=X, y=y)


class TestTransformerInTrainingPipeline(unittest.TestCase):
    """train_models_for_symbol の use_transformer オプションのテスト"""

    @patch("src.prediction.training_pipeline.save_model_metrics")
    @patch("src.prediction.training_pipeline.ModelManager")
    @patch("src.prediction.training_pipeline.load_features_for_training")
    def test_use_transformer_false_trains_two_models(self, mock_load, mock_mm_cls, mock_save):
        """use_transformer=False (デフォルト) では XGBoost + LightGBM の 2 モデルのみ学習されること"""
        from src.prediction.training_pipeline import train_models_for_symbol

        mock_load.return_value = _make_feature_result()
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        mock_model = MagicMock()
        mock_model.predict.return_value = pd.Series(np.zeros(100), index=dates)
        mock_mm = MagicMock()
        mock_mm.get_model.return_value = mock_model
        mock_mm_cls.return_value = mock_mm

        result = train_models_for_symbol("us", "AAPL", use_transformer=False)

        self.assertEqual(result["status"], "success")
        self.assertEqual(mock_mm.create_model.call_count, 2)

    @patch("src.prediction.training_pipeline.save_model_metrics")
    @patch("src.prediction.training_pipeline.ModelManager")
    @patch("src.prediction.training_pipeline.load_features_for_training")
    def test_use_transformer_true_trains_three_models(self, mock_load, mock_mm_cls, mock_save):
        """use_transformer=True のとき XGBoost + LightGBM + TransformerModel の 3 モデルが学習されること"""
        from src.prediction.training_pipeline import train_models_for_symbol

        mock_load.return_value = _make_feature_result()
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        mock_model = MagicMock()
        mock_model.predict.return_value = pd.Series(np.zeros(100), index=dates)
        mock_mm = MagicMock()
        mock_mm.get_model.return_value = mock_model
        mock_mm_cls.return_value = mock_mm

        result = train_models_for_symbol("us", "AAPL", use_transformer=True)

        self.assertEqual(result["status"], "success")
        self.assertEqual(mock_mm.create_model.call_count, 3)
        created_types = [call[0][0] for call in mock_mm.create_model.call_args_list]
        self.assertIn("TransformerModel", created_types)

    @patch("src.prediction.training_pipeline.save_model_metrics")
    @patch("src.prediction.training_pipeline.ModelManager")
    @patch("src.prediction.training_pipeline.load_features_for_training")
    def test_use_transformer_true_uses_transformer_name_prefix(
        self, mock_load, mock_mm_cls, mock_save
    ):
        """use_transformer=True かつ shadow_mode=True のとき ChallengerTransformerModel が作成されること"""
        from src.prediction.training_pipeline import train_models_for_symbol

        mock_load.return_value = _make_feature_result()
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        mock_model = MagicMock()
        mock_model.predict.return_value = pd.Series(np.zeros(100), index=dates)
        mock_mm = MagicMock()
        mock_mm.get_model.return_value = mock_model
        mock_mm_cls.return_value = mock_mm

        train_models_for_symbol("us", "AAPL", shadow_mode=True, use_transformer=True)

        created_names = [call[0][1] for call in mock_mm.create_model.call_args_list]
        self.assertIn("ChallengerTransformerModel", created_names)

    @patch("src.prediction.training_pipeline.train_models_for_symbol")
    def test_task_wrapper_passes_use_transformer(self, mock_train):
        """train_models_for_symbol_task が use_transformer を伝搬すること"""
        from src.prediction.training_pipeline import train_models_for_symbol_task
        from src.watchlist.types import SymbolTask

        mock_train.return_value = {"status": "success", "market": "us", "symbol": "AAPL"}
        task = SymbolTask(market="us", symbol="AAPL", horizon=1)

        train_models_for_symbol_task(task, use_transformer=True)

        mock_train.assert_called_once_with("us", "AAPL", 1, shadow_mode=False, use_transformer=True)


class TestTransformerInUnifiedPipeline(unittest.TestCase):
    """train_unified_models_batch の include_transformer オプションのテスト"""

    @patch("src.prediction.unified_model_pipeline.train_unified_model")
    def test_both_true_without_transformer_trains_two_model_types(self, mock_train):
        """both=True かつ include_transformer=False（デフォルト）では 2 モデルタイプが学習されること"""
        from src.prediction.unified_model_pipeline import train_unified_models_batch

        train_unified_models_batch(horizons=[1], both=True, include_transformer=False)

        self.assertEqual(mock_train.call_count, 2)
        trained_types = [call[1]["model_type"] for call in mock_train.call_args_list]
        self.assertNotIn("TransformerModel", trained_types)

    @patch("src.prediction.unified_model_pipeline.train_unified_model")
    def test_both_true_with_transformer_trains_three_model_types(self, mock_train):
        """both=True かつ include_transformer=True のとき TransformerModel も学習されること"""
        from src.prediction.unified_model_pipeline import train_unified_models_batch

        train_unified_models_batch(horizons=[1], both=True, include_transformer=True)

        self.assertEqual(mock_train.call_count, 3)
        trained_types = [call[1]["model_type"] for call in mock_train.call_args_list]
        self.assertIn("TransformerModel", trained_types)

    @patch("src.prediction.unified_model_pipeline.train_unified_model")
    def test_transformer_model_name_follows_naming_convention(self, mock_train):
        """TransformerModel の統合モデル名が命名規則に従うこと"""
        from src.prediction.unified_model_pipeline import (
            make_unified_model_name,
            train_unified_models_batch,
        )

        train_unified_models_batch(horizons=[1], both=True, include_transformer=True)

        trained_names = [call[1]["model_name"] for call in mock_train.call_args_list]
        expected_name = make_unified_model_name("TransformerModel", 1)
        self.assertIn(expected_name, trained_names)
        self.assertEqual(expected_name, "UnifiedStockTransformer")

    @patch("src.prediction.unified_model_pipeline.train_unified_model")
    def test_include_transformer_false_default_does_not_break_existing(self, mock_train):
        """既存のデフォルト呼び出しが壊れないこと"""
        from src.prediction.unified_model_pipeline import train_unified_models_batch

        train_unified_models_batch(horizons=[1, 3], both=True)

        # horizon × 2 モデルタイプ = 4 呼び出し（TransformerModel なし）
        self.assertEqual(mock_train.call_count, 4)


class TestTransformerInShadowEvaluation(unittest.TestCase):
    """shadow_evaluation の Transformer 統合テスト"""

    def test_unified_production_names_include_transformer(self):
        """_UNIFIED_PRODUCTION_NAMES に UnifiedStockTransformer が含まれること"""
        from src.prediction.shadow_evaluation import _UNIFIED_PRODUCTION_NAMES

        self.assertIn("UnifiedStockTransformer", _UNIFIED_PRODUCTION_NAMES)

    def test_unified_challenger_names_include_transformer(self):
        """_UNIFIED_CHALLENGER_NAMES に UnifiedStockTransformer_challenger が含まれること"""
        from src.prediction.shadow_evaluation import _UNIFIED_CHALLENGER_NAMES

        self.assertIn("UnifiedStockTransformer_challenger", _UNIFIED_CHALLENGER_NAMES)

    def test_predict_with_challenger_returns_empty_when_no_models_exist(self):
        """challenger ファイルが 1 つも存在しない場合は空リストが返ること"""
        from src.prediction.shadow_evaluation import predict_with_challenger_unified

        with patch("os.path.exists", return_value=False):
            result = predict_with_challenger_unified()

        self.assertEqual(result, [])

    def test_predict_with_challenger_uses_available_models_when_transformer_missing(self):
        """TransformerModel challenger が欠損しても XGBoost/LightGBM challenger が存在すれば処理すること"""
        import os

        xgb_chal = "UnifiedStockXGBoost_challenger"
        lgb_chal = "UnifiedStockLightGBM_challenger"

        def _mock_exists(path):
            name = os.path.basename(path).replace(".joblib", "")
            return name in (xgb_chal, lgb_chal)

        with (
            patch("src.utils.data_path_utils.get_models_dir", return_value="/tmp/models"),
            patch("os.path.exists", side_effect=_mock_exists),
            patch(
                "src.prediction.predict_unified.get_cached_model",
                return_value=None,
            ),
            patch(
                "src.utils.db.get_all_symbols",
                return_value=[],
            ),
        ):
            from src.prediction.shadow_evaluation import predict_with_challenger_unified

            result = predict_with_challenger_unified()

        # 空銘柄リストなので空リストが返るが、例外は発生しないこと
        self.assertIsInstance(result, list)


class TestMakeUnifiedModelNameForTransformer(unittest.TestCase):
    """make_unified_model_name のTransformerModel テスト"""

    def test_transformer_model_name_horizon_1(self):
        from src.prediction.unified_model_pipeline import make_unified_model_name

        name = make_unified_model_name("TransformerModel", 1)
        self.assertEqual(name, "UnifiedStockTransformer")

    def test_transformer_model_name_horizon_3(self):
        from src.prediction.unified_model_pipeline import make_unified_model_name

        name = make_unified_model_name("TransformerModel", 3)
        self.assertEqual(name, "UnifiedStockTransformer_3d")


if __name__ == "__main__":
    unittest.main()
