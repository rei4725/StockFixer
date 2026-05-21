"""ユニットテスト: シャドーモード A/B テスト評価パイプライン"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd


def _make_accuracy_df(
    model_name: str, hit_values: list, pred_ratios: list, actual_ratios: list
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_name": [model_name] * len(hit_values),
            "direction_match": hit_values,
            "predicted_ratio": pred_ratios,
            "actual_ratio": actual_ratios,
        }
    )


class TestComputeHitRate(unittest.TestCase):
    def test_empty_df_returns_none(self):
        from src.prediction.shadow_evaluation import _compute_hit_rate

        self.assertIsNone(_compute_hit_rate(pd.DataFrame()))

    def test_missing_column_returns_none(self):
        from src.prediction.shadow_evaluation import _compute_hit_rate

        df = pd.DataFrame({"model_name": ["x"]})
        self.assertIsNone(_compute_hit_rate(df))

    def test_returns_mean_of_direction_match(self):
        from src.prediction.shadow_evaluation import _compute_hit_rate

        df = pd.DataFrame({"direction_match": [1, 1, 0, 1]})
        result = _compute_hit_rate(df)
        self.assertAlmostEqual(result, 0.75, places=5)

    def test_all_correct_returns_one(self):
        from src.prediction.shadow_evaluation import _compute_hit_rate

        df = pd.DataFrame({"direction_match": [1, 1, 1]})
        self.assertAlmostEqual(_compute_hit_rate(df), 1.0, places=5)


class TestComputeSharpe(unittest.TestCase):
    def test_empty_df_returns_none(self):
        from src.prediction.shadow_evaluation import _compute_sharpe

        self.assertIsNone(_compute_sharpe(pd.DataFrame()))

    def test_missing_columns_returns_none(self):
        from src.prediction.shadow_evaluation import _compute_sharpe

        df = pd.DataFrame({"direction_match": [1, 0]})
        self.assertIsNone(_compute_sharpe(df))

    def test_single_row_returns_none(self):
        from src.prediction.shadow_evaluation import _compute_sharpe

        df = pd.DataFrame({"predicted_ratio": [0.01], "actual_ratio": [0.01]})
        self.assertIsNone(_compute_sharpe(df))

    def test_valid_returns_float(self):
        from src.prediction.shadow_evaluation import _compute_sharpe

        df = pd.DataFrame(
            {
                "predicted_ratio": [0.01, 0.02, -0.01, 0.03, -0.02],
                "actual_ratio": [0.01, 0.015, -0.005, 0.025, -0.01],
            }
        )
        result = _compute_sharpe(df)
        self.assertIsInstance(result, float)

    def test_zero_std_returns_none(self):
        from src.prediction.shadow_evaluation import _compute_sharpe

        df = pd.DataFrame(
            {
                "predicted_ratio": [0.01, 0.01, 0.01],
                "actual_ratio": [0.005, 0.005, 0.005],  # 全て同じ → std=0
            }
        )
        result = _compute_sharpe(df)
        self.assertIsNone(result)


class TestEvaluateShadowModels(unittest.TestCase):
    def _make_mock_df(self, model: str) -> pd.DataFrame:
        return _make_accuracy_df(
            model, [1, 1, 0, 1], [0.01, 0.02, -0.01, 0.03], [0.01, 0.015, -0.005, 0.025]
        )

    def test_returns_expected_keys(self):
        from src.prediction.shadow_evaluation import evaluate_shadow_models

        prod_df = self._make_mock_df("production")
        chal_df = pd.concat([self._make_mock_df("challenger"), self._make_mock_df("production")])

        with (
            patch(
                "src.prediction.shadow_evaluation.load_prediction_accuracy",
                side_effect=[prod_df, chal_df],
            ),
            patch("src.prediction.shadow_evaluation._record_ab_result"),
        ):
            result = evaluate_shadow_models()

        for key in (
            "production_hit_rate",
            "production_sharpe",
            "challenger_hit_rate",
            "challenger_sharpe",
            "challenger_wins",
            "evaluated_at",
            "n_production",
            "n_challenger",
        ):
            self.assertIn(key, result)

    def test_challenger_wins_when_both_better(self):
        from src.prediction.shadow_evaluation import evaluate_shadow_models

        # prod: hit=50%, sharpe with non-zero std
        prod_df = _make_accuracy_df(
            "production", [1, 0, 1, 0], [0.01, -0.01, 0.02, -0.02], [0.005, -0.003, 0.015, -0.01]
        )
        # chal: hit=100%, sharpe definitely higher
        chal_df = _make_accuracy_df(
            "challenger", [1, 1, 1, 1], [0.02, 0.03, 0.02, 0.03], [0.02, 0.025, 0.02, 0.028]
        )

        with (
            patch(
                "src.prediction.shadow_evaluation.load_prediction_accuracy",
                side_effect=[prod_df, chal_df],
            ),
            patch("src.prediction.shadow_evaluation._record_ab_result"),
        ):
            result = evaluate_shadow_models()

        # challenger should have better hit_rate
        self.assertGreater(result["challenger_hit_rate"], result["production_hit_rate"])

    def test_empty_dfs_challenger_does_not_win(self):
        from src.prediction.shadow_evaluation import evaluate_shadow_models

        with (
            patch(
                "src.prediction.shadow_evaluation.load_prediction_accuracy",
                return_value=pd.DataFrame(),
            ),
            patch("src.prediction.shadow_evaluation._record_ab_result"),
        ):
            result = evaluate_shadow_models()

        self.assertFalse(result["challenger_wins"])


class TestPromoteUnifiedChallenger(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.unified_dir = os.path.join(self.tmpdir, "unified")
        os.makedirs(self.unified_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_challenger_files(self):
        for name in ["UnifiedStockXGBoost_challenger", "UnifiedStockLightGBM_challenger"]:
            path = os.path.join(self.unified_dir, f"{name}.joblib")
            with open(path, "wb") as f:
                f.write(b"dummy model")

    def test_skips_when_challenger_not_found(self):
        from src.prediction.shadow_evaluation import (
            _UNIFIED_CHALLENGER_NAMES,
            promote_unified_challenger,
        )

        with patch("src.utils.data_path_utils.get_models_dir", return_value=self.tmpdir):
            result = promote_unified_challenger()

        self.assertEqual(len(result["skipped"]), len(_UNIFIED_CHALLENGER_NAMES))
        self.assertEqual(len(result["promoted"]), 0)

    def test_dry_run_does_not_copy_files(self):
        from src.prediction.shadow_evaluation import promote_unified_challenger

        self._create_challenger_files()
        with patch("src.utils.data_path_utils.get_models_dir", return_value=self.tmpdir):
            result = promote_unified_challenger(dry_run=True)

        self.assertEqual(result["dry_run"], True)
        self.assertEqual(len(result["promoted"]), 2)
        # 実際のコピーは行われていない
        for name in ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]:
            self.assertFalse(os.path.exists(os.path.join(self.unified_dir, f"{name}.joblib")))

    def test_promotes_files_when_challenger_exists(self):
        from src.prediction.shadow_evaluation import promote_unified_challenger

        self._create_challenger_files()
        with patch("src.utils.data_path_utils.get_models_dir", return_value=self.tmpdir):
            result = promote_unified_challenger(dry_run=False)

        self.assertEqual(len(result["promoted"]), 2)
        for name in ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]:
            self.assertTrue(os.path.exists(os.path.join(self.unified_dir, f"{name}.joblib")))


class TestPromoteChallengerToProduction(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_skips_when_no_challenger_files(self):
        from src.prediction.shadow_evaluation import promote_challenger_to_production

        with patch("src.prediction.shadow_evaluation.get_models_subdir", return_value=self.tmpdir):
            result = promote_challenger_to_production("jp", "7203")

        self.assertEqual(len(result["skipped"]), 2)
        self.assertEqual(len(result["promoted"]), 0)

    def test_dry_run_logs_but_does_not_copy(self):
        from src.prediction.shadow_evaluation import promote_challenger_to_production

        for name in ["ChallengerXGBoostModel", "ChallengerLightGBMModel"]:
            with open(os.path.join(self.tmpdir, f"{name}.joblib"), "wb") as f:
                f.write(b"dummy")

        with patch("src.prediction.shadow_evaluation.get_models_subdir", return_value=self.tmpdir):
            result = promote_challenger_to_production("jp", "7203", dry_run=True)

        self.assertEqual(result["dry_run"], True)
        self.assertEqual(len(result["promoted"]), 2)
        for name in ["StockXGBoostModel", "StockLightGBMModel"]:
            self.assertFalse(os.path.exists(os.path.join(self.tmpdir, f"{name}.joblib")))


class TestRunShadowPrediction(unittest.TestCase):
    def test_returns_counts(self):
        from src.prediction.shadow_evaluation import run_shadow_prediction

        mock_prod_fn = MagicMock(return_value=["r1", "r2"])
        mock_chal_fn = MagicMock(return_value=["r3"])

        with patch("src.prediction.prediction_pipeline.output_top_worst_results"):
            result = run_shadow_prediction(mock_prod_fn, challenger_predict_fn=mock_chal_fn)

        self.assertEqual(result["production_count"], 2)
        self.assertEqual(result["challenger_count"], 1)

    def test_uses_prod_fn_as_fallback_for_challenger(self):
        from src.prediction.shadow_evaluation import run_shadow_prediction

        mock_fn = MagicMock(return_value=["r1"])

        with patch("src.prediction.prediction_pipeline.output_top_worst_results"):
            run_shadow_prediction(mock_fn)

        self.assertEqual(mock_fn.call_count, 2)


if __name__ == "__main__":
    unittest.main()
