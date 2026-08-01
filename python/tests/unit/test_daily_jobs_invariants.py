"""日次パイプラインへの出力 invariant 配線のテスト。"""

import unittest
from unittest.mock import MagicMock, patch

from src.prediction.types import PredictionResult
from src.utils.alert_service import AlertResult


def _rows(count: int, model_count: int):
    return [
        PredictionResult(
            market="jp",
            symbol=str(7000 + i),
            current_price=1000.0 + i,
            avg_pred_price=1010.0 + i,
            diff_ratio=0.001 * (i - count / 2),
            model_count=model_count,
        )
        for i in range(count)
    ]


class TestDailyPipelineInvariantWiring(unittest.TestCase):
    def _run_pipeline(self, output_rows, loaded_models, previous, predict_side_effect=None):
        """日次パイプラインを最小のモックで走らせ、通知に渡った results を返す。"""
        captured = {}

        def fake_run_conditional_notification(results=None, **kwargs):
            captured["results"] = results
            return True

        if predict_side_effect is not None:
            predict_patch = patch(
                "src.prediction.prediction_pipeline.predict_all_unified",
                side_effect=predict_side_effect,
            )
        else:
            predict_patch = patch(
                "src.prediction.prediction_pipeline.predict_all_unified",
                return_value=output_rows,
            )

        with (
            patch("src.watchlist.batch_runner.load_target_symbols", return_value=[]),
            patch("src.market_data.pipeline.run_batch_pipeline"),
            patch(
                "src.infrastructure.discord_notification_adapter." "DiscordNotificationAdapter",
                MagicMock(),
            ),
            patch(
                "src.prediction.predict_unified.preload_models",
                return_value=loaded_models,
            ),
            predict_patch,
            patch("src.prediction.prediction_pipeline.output_top_worst_results"),
            patch(
                "src.prediction.db.prediction_results.load_previous_run_stats",
                return_value=previous,
            ),
            patch(
                "src.prediction.shadow_evaluation.predict_with_challenger_unified", return_value=[]
            ),
            patch("src.prediction.prediction_pipeline.run_accuracy_check", return_value={}),
            patch("src.reporting.discord.discord_utils.send_accuracy_summary"),
            patch("src.orchestration.jobs.daily.run_daily_drift_check"),
            patch("src.reporting.discord.discord_utils.send_daily_pipeline_completion"),
            patch("src.reporting.discord.discord_utils.send_daily_pipeline_error"),
            patch(
                "src.utils.alert_service.run_conditional_notification",
                side_effect=fake_run_conditional_notification,
            ),
            # 既存 4 ルールは DB / 状態ファイルを触るためテストから遮断する。
            # 密閉性が崩れると CI とデプロイが落ちる（#516 / #517 の前例）。
            patch(
                "src.utils.alert_service.check_pipeline_fail_rule",
                return_value=AlertResult("NF-303-1", "A", False, 0, 2),
            ),
            patch(
                "src.utils.alert_service.check_loss_limit_rule",
                return_value=AlertResult("NF-303-2", "B", False, 0, 3),
            ),
            patch(
                "src.utils.alert_service.check_drift_warn_rule",
                return_value=AlertResult("NF-303-3", "C", False, 0, 2),
            ),
            patch(
                "src.utils.alert_service.check_health_degraded_rule",
                return_value=AlertResult("NF-303-4", "D", False, 0, 2),
            ),
        ):
            from src.orchestration.jobs.daily import run_daily_pipeline

            run_daily_pipeline()

        return captured.get("results")

    def test_degraded_ensemble_reaches_notification(self):
        """#615 の状態が NF-303-5 の triggered として通知に届く。"""
        results = self._run_pipeline(
            output_rows=_rows(705, model_count=1),
            loaded_models=["UnifiedStockXGBoost", "UnifiedStockLightGBM"],
            previous=None,
        )
        self.assertIsNotNone(results)
        rule = next(r for r in results if r.rule_id == "NF-303-5")
        self.assertTrue(rule.triggered)
        self.assertIn("A-2", rule.details["violation_ids"])

    def test_healthy_run_is_not_triggered(self):
        results = self._run_pipeline(
            output_rows=_rows(705, model_count=2),
            loaded_models=["UnifiedStockXGBoost", "UnifiedStockLightGBM"],
            previous=None,
        )
        rule = next(r for r in results if r.rule_id == "NF-303-5")
        self.assertFalse(rule.triggered)

    def test_prediction_stage_failure_propagates_without_notifying(self):
        """[2/5] が CRITICAL 失敗した場合は例外が伝播し、通知段まで到達しない。

        _handle_stage_error(PipelineStage.CRITICAL, ...) は常に True を返す
        （src/orchestration/jobs/common.py 参照）ため、run_daily_pipeline() は
        [2/5] の例外をそのまま re-raise する。[2.1/5]・[6/6] には到達しないので
        NF-303-5 の A-0/A-3 分岐そのものはここでは検証しない
        （その分岐は predict_all_unified が None を握りつぶす別経路で発生する）。
        """
        with self.assertRaises(RuntimeError):
            self._run_pipeline(
                output_rows=None,
                loaded_models=["UnifiedStockXGBoost", "UnifiedStockLightGBM"],
                previous=None,
                predict_side_effect=RuntimeError("boom"),
            )


if __name__ == "__main__":
    unittest.main()
