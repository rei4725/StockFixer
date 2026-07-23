"""ユニットテスト: モデル昇格ゲート（evaluate_promotion / save_promotion_result）"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd


def _make_wf_df(model_name: str | None = None) -> pd.DataFrame:
    """Walk-Forward サマリー風 DataFrame を生成する。"""
    data: dict[str, list] = {
        "total_return": [0.05, 0.06],
        "max_drawdown": [-0.10, -0.08],
        "sharpe_ratio": [0.5, 0.6],
    }
    if model_name:
        data["model_name"] = [model_name, model_name]
    return pd.DataFrame(data)


def _make_accuracy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "direction_match": [1, 1, 0, 1],
            "predicted_ratio": [0.01, 0.02, -0.01, 0.03],
            "actual_ratio": [0.01, 0.015, -0.005, 0.025],
            "model_name": ["shadow", "shadow", "shadow", "shadow"],
        }
    )


class TestWfMean(unittest.TestCase):
    def test_returns_mean_for_model_name_column(self):
        from src.prediction.promotion_gate import _wf_mean

        df = pd.DataFrame(
            {
                "model_name": ["shadow", "current", "shadow"],
                "total_return": [0.05, 0.03, 0.07],
            }
        )
        result = _wf_mean(df, "shadow", "total_return")
        self.assertAlmostEqual(result, 0.06, places=5)

    def test_returns_mean_when_no_model_name_column(self):
        from src.prediction.promotion_gate import _wf_mean

        df = pd.DataFrame({"total_return": [0.04, 0.06]})
        result = _wf_mean(df, "shadow", "total_return")
        self.assertAlmostEqual(result, 0.05, places=5)

    def test_returns_none_when_column_missing(self):
        from src.prediction.promotion_gate import _wf_mean

        df = pd.DataFrame({"model_name": ["shadow"], "total_return": [0.05]})
        result = _wf_mean(df, "shadow", "nonexistent_col")
        self.assertIsNone(result)

    def test_returns_none_when_no_matching_rows(self):
        from src.prediction.promotion_gate import _wf_mean

        df = pd.DataFrame({"model_name": ["current"], "total_return": [0.05]})
        result = _wf_mean(df, "shadow", "total_return")
        self.assertIsNone(result)

    def test_empty_df_returns_none(self):
        from src.prediction.promotion_gate import _wf_mean

        result = _wf_mean(pd.DataFrame(), "shadow", "total_return")
        self.assertIsNone(result)


class TestComputeHitRate(unittest.TestCase):
    def test_returns_hit_rate(self):
        from src.prediction.promotion_gate import _compute_hit_rate

        df = _make_accuracy_df()
        with patch("src.prediction.promotion_gate.load_prediction_accuracy", return_value=df):
            result = _compute_hit_rate("shadow", horizon=1)
        self.assertAlmostEqual(result, 0.75, places=5)

    def test_returns_none_when_empty_df(self):
        from src.prediction.promotion_gate import _compute_hit_rate

        with patch(
            "src.prediction.promotion_gate.load_prediction_accuracy",
            return_value=pd.DataFrame(),
        ):
            result = _compute_hit_rate("shadow", horizon=1)
        self.assertIsNone(result)

    def test_returns_none_when_no_direction_match_column(self):
        from src.prediction.promotion_gate import _compute_hit_rate

        df = pd.DataFrame({"model_name": ["shadow"], "predicted_ratio": [0.01]})
        with patch("src.prediction.promotion_gate.load_prediction_accuracy", return_value=df):
            result = _compute_hit_rate("shadow", horizon=1)
        self.assertIsNone(result)

    def test_filters_by_model_name(self):
        from src.prediction.promotion_gate import _compute_hit_rate

        df = pd.DataFrame(
            {
                "model_name": ["shadow", "other"],
                "direction_match": [1, 0],
            }
        )
        with patch("src.prediction.promotion_gate.load_prediction_accuracy", return_value=df):
            result = _compute_hit_rate("shadow", horizon=1)
        self.assertAlmostEqual(result, 1.0, places=5)


class TestComputeSlippage(unittest.TestCase):
    def test_returns_slippage(self):
        from src.prediction.promotion_gate import _compute_slippage

        df = _make_accuracy_df()
        with patch("src.prediction.promotion_gate.load_prediction_accuracy", return_value=df):
            result = _compute_slippage("shadow", horizon=1)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result, 0.0)

    def test_returns_none_when_empty(self):
        from src.prediction.promotion_gate import _compute_slippage

        with patch(
            "src.prediction.promotion_gate.load_prediction_accuracy",
            return_value=pd.DataFrame(),
        ):
            result = _compute_slippage("shadow", horizon=1)
        self.assertIsNone(result)

    def test_returns_none_when_missing_columns(self):
        from src.prediction.promotion_gate import _compute_slippage

        df = pd.DataFrame({"model_name": ["shadow"], "direction_match": [1]})
        with patch("src.prediction.promotion_gate.load_prediction_accuracy", return_value=df):
            result = _compute_slippage("shadow", horizon=1)
        self.assertIsNone(result)


class TestCollectRunIds(unittest.TestCase):
    def test_collects_run_ids(self):
        from src.prediction.promotion_gate import _collect_run_ids

        mock_df = pd.DataFrame({"run_id": ["r1", "r2", "r3"]})
        with patch("src.prediction.promotion_gate.load_experiment_runs", return_value=mock_df):
            result = _collect_run_ids("shadow", "current")
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_empty_df_returns_empty(self):
        from src.prediction.promotion_gate import _collect_run_ids

        with patch(
            "src.prediction.promotion_gate.load_experiment_runs", return_value=pd.DataFrame()
        ):
            result = _collect_run_ids("shadow", "current")
        self.assertEqual(result, [])


class TestEvaluatePromotion(unittest.TestCase):
    def _patch_all(self, wf_df=None, hit_rate=0.6, slippage=0.01, run_ids=None):
        if wf_df is None:
            wf_df = pd.DataFrame(
                {
                    "model_name": ["shadow", "current"],
                    "total_return": [0.06, 0.04],
                    "max_drawdown": [-0.08, -0.12],
                    "sharpe_ratio": [0.7, 0.6],
                }
            )
        if run_ids is None:
            run_ids = ["r1", "r2"]
        return [
            patch(
                "src.prediction.promotion_gate._load_latest_wf_summary",
                return_value=(wf_df, "wf_summary_20240101.csv"),
            ),
            patch("src.prediction.promotion_gate._compute_hit_rate", return_value=hit_rate),
            patch("src.prediction.promotion_gate._compute_slippage", return_value=slippage),
            patch("src.prediction.promotion_gate._collect_run_ids", return_value=run_ids),
        ]

    def test_eligible_when_all_criteria_met(self):
        from src.prediction.promotion_gate import evaluate_promotion

        patches = self._patch_all()
        with patches[0], patches[1], patches[2], patches[3]:
            result = evaluate_promotion("shadow", "current")
        self.assertTrue(result.eligible)
        self.assertEqual(result.shadow_model, "shadow")
        self.assertEqual(result.current_model, "current")

    def test_not_eligible_when_hit_rate_below_floor(self):
        from src.prediction.promotion_gate import evaluate_promotion

        patches = self._patch_all(hit_rate=0.4)
        with patches[0], patches[1], patches[2], patches[3]:
            result = evaluate_promotion("shadow", "current")
        self.assertFalse(result.eligible)
        self.assertIn("hit_rate", result.criteria)

    def test_not_eligible_when_slippage_too_high(self):
        from src.prediction.promotion_gate import evaluate_promotion

        patches = self._patch_all(slippage=0.05)
        with patches[0], patches[1], patches[2], patches[3]:
            result = evaluate_promotion("shadow", "current")
        self.assertFalse(result.eligible)

    def test_no_wf_data_returns_not_eligible(self):
        from src.prediction.promotion_gate import evaluate_promotion

        with (
            patch(
                "src.prediction.promotion_gate._load_latest_wf_summary",
                return_value=(None, None),
            ),
            patch("src.prediction.promotion_gate._compute_hit_rate", return_value=0.6),
            patch("src.prediction.promotion_gate._compute_slippage", return_value=0.01),
            patch("src.prediction.promotion_gate._collect_run_ids", return_value=[]),
        ):
            result = evaluate_promotion("shadow", "current")
        self.assertFalse(result.eligible)

    def test_manual_approval_in_reason_when_eligible(self):
        from src.prediction.promotion_gate import evaluate_promotion

        patches = self._patch_all()
        with patches[0], patches[1], patches[2], patches[3]:
            result = evaluate_promotion("shadow", "current", require_manual_approval=True)
        self.assertIn("手動承認", result.reason)

    def test_auto_promote_in_reason_when_no_approval(self):
        from src.prediction.promotion_gate import evaluate_promotion

        patches = self._patch_all()
        with patches[0], patches[1], patches[2], patches[3]:
            result = evaluate_promotion("shadow", "current", require_manual_approval=False)
        self.assertIn("自動昇格", result.reason)

    def test_shadow_run_id_added_to_run_ids(self):
        from src.prediction.promotion_gate import evaluate_promotion

        patches = self._patch_all(run_ids=["existing"])
        with patches[0], patches[1], patches[2], patches[3]:
            result = evaluate_promotion("shadow", "current", shadow_run_id="new_shadow_run")
        self.assertIn("new_shadow_run", result.run_ids)

    def test_criteria_keys_present(self):
        from src.prediction.promotion_gate import evaluate_promotion

        patches = self._patch_all()
        with patches[0], patches[1], patches[2], patches[3]:
            result = evaluate_promotion("shadow", "current")
        for key in ("net_return", "mdd", "sharpe", "hit_rate", "slippage"):
            self.assertIn(key, result.criteria)

    def test_returns_promotion_result_type(self):
        from src.prediction.promotion_gate import PromotionResult, evaluate_promotion

        patches = self._patch_all()
        with patches[0], patches[1], patches[2], patches[3]:
            result = evaluate_promotion("shadow", "current")
        self.assertIsInstance(result, PromotionResult)


class TestSavePromotionResult(unittest.TestCase):
    def test_saves_json_file(self):
        from src.prediction.promotion_gate import PromotionResult, save_promotion_result

        result = PromotionResult(
            shadow_model="shadow",
            current_model="current",
            evaluated_at="2024-01-01T00:00:00",
            eligible=True,
            require_manual_approval=True,
            criteria={"net_return": {"shadow": 0.06, "current": 0.04, "passed": True}},
            reason="全基準クリア",
            run_ids=["r1"],
            wf_source_file="wf_summary_20240101.csv",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.prediction.promotion_gate.get_results_dir", return_value=tmpdir):
                path = save_promotion_result(result)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        self.assertEqual(data["eligible"], True)
        self.assertEqual(data["shadow_model"], "shadow")


class TestAutoPromoteModelSetting(unittest.TestCase):
    def test_default_is_false(self):
        """AUTO_PROMOTE_MODEL はデフォルト False。"""
        with patch.dict("os.environ", {}, clear=False):
            import importlib

            import config.feature_flags as feature_flags_mod
            import config.settings as settings_mod

            # AUTO_PROMOTE_MODEL は config.feature_flags.flags 側にある。
            # settings_mod は `from config.feature_flags import flags` で束縛して
            # いるだけなので、settings_mod を reload しても feature_flags_mod 側を
            # 先に reload しないと古い flags インスタンスを再束縛するだけになる。
            importlib.reload(feature_flags_mod)
            importlib.reload(settings_mod)
            self.assertFalse(settings_mod.AUTO_PROMOTE_MODEL)

    def test_env_true_sets_true(self):
        """AUTO_PROMOTE_MODEL=true が True に変換される。"""
        with patch.dict("os.environ", {"AUTO_PROMOTE_MODEL": "true"}, clear=False):
            import importlib

            import config.feature_flags as feature_flags_mod
            import config.settings as settings_mod

            importlib.reload(feature_flags_mod)
            importlib.reload(settings_mod)
            self.assertTrue(settings_mod.AUTO_PROMOTE_MODEL)


class TestSchedulerAutoPromotion(unittest.TestCase):
    """スケジューラの AUTO_PROMOTE_MODEL による昇格制御のテスト。"""

    def _make_patches(self, auto_promote: bool, eligible: bool = True):
        wf_df = pd.DataFrame(
            {
                "model_name": ["challenger", "production"],
                "total_return": [0.06, 0.04],
                "max_drawdown": [-0.08, -0.12],
                "sharpe_ratio": [0.7, 0.6],
            }
        )
        return [
            patch("config.settings.AUTO_PROMOTE_MODEL", auto_promote),
            patch(
                "src.prediction.promotion_gate._load_latest_wf_summary",
                return_value=(wf_df, "wf_summary.csv"),
            ),
            patch(
                "src.prediction.promotion_gate._compute_hit_rate",
                return_value=0.6 if eligible else 0.3,
            ),
            patch("src.prediction.promotion_gate._compute_slippage", return_value=0.01),
            patch("src.prediction.promotion_gate._collect_run_ids", return_value=[]),
        ]

    def test_require_manual_approval_false_when_auto_promote_true(self):
        """AUTO_PROMOTE_MODEL=True → require_manual_approval=False が渡される。"""
        from src.prediction.promotion_gate import evaluate_promotion

        patches = self._make_patches(auto_promote=True)
        with patches[1], patches[2], patches[3], patches[4]:
            result = evaluate_promotion("challenger", "production", require_manual_approval=False)
        self.assertFalse(result.require_manual_approval)
        self.assertIn("自動昇格", result.reason)

    def test_require_manual_approval_true_when_auto_promote_false(self):
        """AUTO_PROMOTE_MODEL=False → require_manual_approval=True が渡される。"""
        from src.prediction.promotion_gate import evaluate_promotion

        patches = self._make_patches(auto_promote=False)
        with patches[1], patches[2], patches[3], patches[4]:
            result = evaluate_promotion("challenger", "production", require_manual_approval=True)
        self.assertTrue(result.require_manual_approval)
        self.assertIn("手動承認", result.reason)

    def test_auto_promote_false_skips_promotion(self):
        """AUTO_PROMOTE_MODEL=False かつ eligible=True でも promote_unified_challenger は呼ばれない。"""
        mock_promote = MagicMock(return_value={"promoted": ["challenger.joblib"], "skipped": []})

        with (
            patch("src.orchestration.scheduler.run_weekly_training.__module__"),
            patch("config.settings.AUTO_PROMOTE_MODEL", False),
            patch(
                "src.prediction.promotion_gate.evaluate_promotion",
            ) as mock_eval,
            patch("src.prediction.promotion_gate.save_promotion_result"),
            patch(
                "src.prediction.shadow_evaluation.evaluate_shadow_models",
                return_value={"challenger_wins": True},
            ),
            patch("src.prediction.shadow_evaluation.promote_unified_challenger", mock_promote),
            patch("src.prediction.shadow_evaluation._UNIFIED_PRODUCTION_NAMES", ["production_xgb"]),
            patch("src.prediction.shadow_evaluation._UNIFIED_CHALLENGER_NAMES", ["challenger_xgb"]),
            patch("src.prediction.unified_model_pipeline.train_unified_model"),
            patch(
                "src.prediction.prediction_pipeline.run_accuracy_check", return_value=pd.DataFrame()
            ),
            patch("src.reporting.discord.discord_utils.send_drift_alert"),
            patch("src.reporting.discord.discord_utils.send_weekly_training_completion"),
            patch("src.reporting.discord.discord_utils.send_promotion_result"),
        ):
            from src.prediction.promotion_gate import PromotionResult

            mock_eval.return_value = PromotionResult(
                shadow_model="challenger",
                current_model="production",
                evaluated_at="2024-01-01T00:00:00",
                eligible=True,
                require_manual_approval=True,
                criteria={},
                reason="全基準クリア。手動承認待ち。",
            )
            from src.orchestration.scheduler import run_weekly_training

            run_weekly_training()

        mock_promote.assert_not_called()

    def test_auto_promote_true_executes_promotion(self):
        """AUTO_PROMOTE_MODEL=True かつ eligible=True で promote_unified_challenger が呼ばれる。"""
        mock_promote = MagicMock(return_value={"promoted": ["challenger.joblib"], "skipped": []})

        with (
            patch("config.settings.AUTO_PROMOTE_MODEL", True),
            patch(
                "src.prediction.promotion_gate.evaluate_promotion",
            ) as mock_eval,
            patch("src.prediction.promotion_gate.save_promotion_result"),
            patch(
                "src.prediction.shadow_evaluation.evaluate_shadow_models",
                return_value={"challenger_wins": True},
            ),
            patch("src.prediction.shadow_evaluation.promote_unified_challenger", mock_promote),
            patch("src.prediction.shadow_evaluation._UNIFIED_PRODUCTION_NAMES", ["production_xgb"]),
            patch("src.prediction.shadow_evaluation._UNIFIED_CHALLENGER_NAMES", ["challenger_xgb"]),
            patch("src.prediction.unified_model_pipeline.train_unified_model"),
            patch(
                "src.prediction.prediction_pipeline.run_accuracy_check", return_value=pd.DataFrame()
            ),
            patch("src.reporting.discord.discord_utils.send_drift_alert"),
            patch("src.reporting.discord.discord_utils.send_weekly_training_completion"),
            patch("src.reporting.discord.discord_utils.send_promotion_result"),
        ):
            from src.prediction.promotion_gate import PromotionResult

            mock_eval.return_value = PromotionResult(
                shadow_model="challenger",
                current_model="production",
                evaluated_at="2024-01-01T00:00:00",
                eligible=True,
                require_manual_approval=False,
                criteria={},
                reason="全基準クリア。自動昇格可能。",
            )
            from src.orchestration.scheduler import run_weekly_training

            run_weekly_training()

        mock_promote.assert_called_once()


if __name__ == "__main__":
    unittest.main()
