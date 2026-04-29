"""model_training_pipeline の純粋関数テスト"""

import unittest

import pandas as pd


class TestComputeTrainingMetrics(unittest.TestCase):
    """_compute_training_metrics のテスト（純粋関数）"""

    def _fn(self, y_true, y_pred):
        from src.services.model_training_pipeline import _compute_training_metrics

        return _compute_training_metrics(
            pd.Series(y_true, dtype=float),
            pd.Series(y_pred, dtype=float),
        )

    def test_returns_training_metrics(self):
        from src.prediction.types import TrainingMetrics

        result = self._fn([0.01, 0.02, -0.01], [0.01, 0.02, -0.01])
        self.assertIsInstance(result, TrainingMetrics)

    def test_perfect_prediction_rmse_zero(self):
        result = self._fn([0.01, 0.02, -0.01], [0.01, 0.02, -0.01])
        self.assertAlmostEqual(result.rmse, 0.0)

    def test_perfect_prediction_directional_accuracy_one(self):
        result = self._fn([0.01, 0.02, -0.01], [0.01, 0.02, -0.01])
        self.assertAlmostEqual(result.directional_accuracy, 1.0)

    def test_all_wrong_direction_accuracy_zero(self):
        result = self._fn([0.01, 0.02, 0.03], [-0.01, -0.02, -0.03])
        self.assertAlmostEqual(result.directional_accuracy, 0.0)

    def test_rmse_non_zero_on_errors(self):
        result = self._fn([0.0, 0.0], [0.1, 0.1])
        self.assertGreater(result.rmse, 0.0)

    def test_n_samples_correct(self):
        result = self._fn([0.01, 0.02, -0.01, 0.05], [0.01, 0.01, -0.01, 0.04])
        self.assertEqual(result.n_samples, 4)

    def test_partial_direction_accuracy(self):
        """2/4 が正解 → 0.5"""
        result = self._fn([0.01, 0.02, -0.01, -0.02], [0.01, 0.02, 0.01, 0.02])
        self.assertAlmostEqual(result.directional_accuracy, 0.5)

    def test_rmse_symmetric(self):
        """残差が +0.1 でも -0.1 でも RMSE は同じ"""
        r1 = self._fn([0.0], [0.1])
        r2 = self._fn([0.0], [-0.1])
        self.assertAlmostEqual(r1.rmse, r2.rmse)


if __name__ == "__main__":
    unittest.main()
