"""MLflow トラッキング統合のユニットテスト"""

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.prediction.types import TrainingMetrics


def _make_mock_mlflow():
    """context manager を正しく実装した mlflow モック。"""
    mock_mlflow = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-id-123"
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=mock_run)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value = ctx
    return mock_mlflow


class TestLogToMlflow(unittest.TestCase):
    """_log_to_mlflow のテスト"""

    def _make_metrics(self):
        return TrainingMetrics(rmse=0.01, directional_accuracy=0.6, n_samples=100)

    def test_logs_params_and_metrics(self):
        """mlflow に params と metrics が記録されること"""
        from src.prediction.training_pipeline import _log_to_mlflow

        mock_mlflow = _make_mock_mlflow()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with patch("os.path.exists", return_value=True):
                _log_to_mlflow(
                    market="us",
                    symbol="AAPL",
                    model_name="StockXGBoostModel",
                    model_type="XGBoostModel",
                    horizon=1,
                    mode_label="production",
                    run_id="db-run-001",
                    metrics=self._make_metrics(),
                    model_path="/some/path.joblib",
                    hyperparams={"n_estimators": 500},
                )

        mock_mlflow.log_params.assert_called_once()
        logged_params = mock_mlflow.log_params.call_args[0][0]
        self.assertEqual(logged_params["market"], "us")
        self.assertEqual(logged_params["symbol"], "AAPL")
        self.assertIn("hp_n_estimators", logged_params)

        mock_mlflow.log_metrics.assert_called_once()
        logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
        self.assertIn("rmse", logged_metrics)
        self.assertIn("directional_accuracy", logged_metrics)
        self.assertIn("n_samples", logged_metrics)

        mock_mlflow.log_artifact.assert_called_once_with("/some/path.joblib")

    def test_no_exception_on_mlflow_error(self):
        """mlflow エラー時も例外が外に伝播しないこと"""
        from src.prediction.training_pipeline import _log_to_mlflow

        mock_mlflow = MagicMock()
        mock_mlflow.set_experiment.side_effect = Exception("mlflow connection error")

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            _log_to_mlflow(
                market="us",
                symbol="AAPL",
                model_name="StockXGBoostModel",
                model_type="XGBoostModel",
                horizon=1,
                mode_label="production",
                run_id="db-run-001",
                metrics=None,
                model_path=None,
                hyperparams=None,
            )

    def test_skips_artifact_when_path_not_exists(self):
        """モデルファイルが存在しない場合はアーティファクトをスキップすること"""
        from src.prediction.training_pipeline import _log_to_mlflow

        mock_mlflow = _make_mock_mlflow()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with patch("os.path.exists", return_value=False):
                _log_to_mlflow(
                    market="us",
                    symbol="AAPL",
                    model_name="StockXGBoostModel",
                    model_type="XGBoostModel",
                    horizon=1,
                    mode_label="production",
                    run_id="db-run-001",
                    metrics=None,
                    model_path="/nonexistent/path.joblib",
                    hyperparams=None,
                )

        mock_mlflow.log_artifact.assert_not_called()

    def test_skips_metrics_when_none(self):
        """metrics が None のとき log_metrics が呼ばれないこと"""
        from src.prediction.training_pipeline import _log_to_mlflow

        mock_mlflow = _make_mock_mlflow()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with patch("os.path.exists", return_value=False):
                _log_to_mlflow(
                    market="jp",
                    symbol="7203",
                    model_name="StockLightGBMModel",
                    model_type="LightGBMModel",
                    horizon=3,
                    mode_label="challenger",
                    run_id="db-run-002",
                    metrics=None,
                    model_path=None,
                    hyperparams=None,
                )

        mock_mlflow.log_metrics.assert_not_called()

    def test_db_run_id_set_as_tag(self):
        """DB の run_id が MLflow タグに記録されること"""
        from src.prediction.training_pipeline import _log_to_mlflow

        mock_mlflow = _make_mock_mlflow()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with patch("os.path.exists", return_value=False):
                _log_to_mlflow(
                    market="us",
                    symbol="MSFT",
                    model_name="StockXGBoostModel",
                    model_type="XGBoostModel",
                    horizon=1,
                    mode_label="production",
                    run_id="db-run-xyz",
                    metrics=None,
                    model_path=None,
                    hyperparams=None,
                )

        mock_mlflow.set_tag.assert_called_once_with("db_run_id", "db-run-xyz")


class TestLogPromotionToMlflow(unittest.TestCase):
    """_log_promotion_to_mlflow のテスト"""

    def _make_result(self, eligible=True):
        from src.prediction.promotion_gate import PromotionResult

        return PromotionResult(
            shadow_model="ChallengerXGBoostModel",
            current_model="StockXGBoostModel",
            evaluated_at="2026-05-17T12:00:00",
            eligible=eligible,
            require_manual_approval=True,
            criteria={
                "net_return": {"passed": True},
                "mdd": {"passed": True},
                "sharpe": {"passed": True},
                "hit_rate": {"passed": eligible},
                "slippage": {"passed": eligible},
            },
            reason="全基準クリア" if eligible else "昇格基準未達",
            run_ids=["run-001", "run-002"],
        )

    def test_logs_eligible_tag(self):
        """eligible タグが正しく記録されること"""
        from src.prediction.promotion_gate import _log_promotion_to_mlflow

        mock_mlflow = _make_mock_mlflow()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            _log_promotion_to_mlflow(self._make_result(eligible=True))

        mock_mlflow.set_tags.assert_called_once()
        tags = mock_mlflow.set_tags.call_args[0][0]
        self.assertEqual(tags["eligible"], "True")
        self.assertEqual(tags["shadow_model"], "ChallengerXGBoostModel")
        self.assertEqual(tags["current_model"], "StockXGBoostModel")

    def test_logs_criteria_as_metrics(self):
        """各基準が metrics として記録されること"""
        from src.prediction.promotion_gate import _log_promotion_to_mlflow

        mock_mlflow = _make_mock_mlflow()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            _log_promotion_to_mlflow(self._make_result())

        self.assertEqual(mock_mlflow.log_metric.call_count, 5)
        logged_names = [call[0][0] for call in mock_mlflow.log_metric.call_args_list]
        self.assertIn("net_return_passed", logged_names)
        self.assertIn("hit_rate_passed", logged_names)

    def test_no_exception_on_mlflow_error(self):
        """mlflow エラー時も例外が外に伝播しないこと"""
        from src.prediction.promotion_gate import _log_promotion_to_mlflow

        mock_mlflow = MagicMock()
        mock_mlflow.set_experiment.side_effect = Exception("mlflow error")

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            _log_promotion_to_mlflow(self._make_result())

    def test_run_ids_set_as_tag(self):
        """run_ids が MLflow タグに記録されること"""
        from src.prediction.promotion_gate import _log_promotion_to_mlflow

        mock_mlflow = _make_mock_mlflow()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            _log_promotion_to_mlflow(self._make_result())

        tag_calls = {call[0][0]: call[0][1] for call in mock_mlflow.set_tag.call_args_list}
        self.assertIn("db_run_ids", tag_calls)
        self.assertIn("run-001", tag_calls["db_run_ids"])

    def test_ineligible_result_logs_correctly(self):
        """昇格不可の結果でも正しくログが記録されること"""
        from src.prediction.promotion_gate import _log_promotion_to_mlflow

        mock_mlflow = _make_mock_mlflow()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            _log_promotion_to_mlflow(self._make_result(eligible=False))

        mock_mlflow.set_tags.assert_called_once()
        tags = mock_mlflow.set_tags.call_args[0][0]
        self.assertEqual(tags["eligible"], "False")


if __name__ == "__main__":
    unittest.main()
