"""日次パイプラインへの出力 invariant 配線のテスト。"""

import unittest
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

from src.prediction.types import PredictionResult
from src.reporting.discord.webhook_sender import send_webhook_notification
from src.utils.alert_service import AlertResult


def _rows(count: int, model_count: int, diff_ratio_scale: float = 0.001):
    return [
        PredictionResult(
            market="jp",
            symbol=str(7000 + i),
            current_price=1000.0 + i,
            avg_pred_price=1010.0 + i,
            diff_ratio=diff_ratio_scale * (i - count / 2),
            model_count=model_count,
        )
        for i in range(count)
    ]


class TestDailyPipelineInvariantWiring(unittest.TestCase):
    def _run_pipeline(
        self,
        output_rows,
        loaded_models,
        previous_raw,
        predict_side_effect=None,
        latest_timestamp_side_effect=None,
        evaluate_side_effect=None,
    ):
        """日次パイプラインを最小のモックで走らせ、通知に渡った results を返す。

        呼び出しに使われた kwargs（notifier を含む）は self._captured に残す。

        C-1 対策後の [2/5] は「保存前」に前回ラン統計を取得する。具体的には
        `src.prediction.db.load_latest_prediction_timestamp` →
        `src.prediction.db.prediction_results.load_run_stats_at` →
        `build_run_stats`（実関数、非patch）の順で呼ばれ、結果を
        `evaluate_output_invariants` へ previous_stats として渡す。

        previous_raw は load_run_stats_at の戻り値（(model_counts, diff_ratios)
        のタプル、または None）を表す。
        """
        captured = {}
        self._captured = captured

        def fake_run_conditional_notification(results=None, **kwargs):
            captured["results"] = results
            captured["notifier"] = kwargs.get("notifier")
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

        # daily.py の [2/5] は `from src.prediction.db import
        # load_latest_prediction_timestamp` を呼び出し都度ローカル import する。
        # ここを patch しないと実 DB 接続に到達してしまう。既定では真の前回
        # タイムスタンプが存在するケースを模して固定文字列を返し、例外注入
        # テストでは side_effect で上書きする。
        if latest_timestamp_side_effect is not None:
            timestamp_patch = patch(
                "src.prediction.db.load_latest_prediction_timestamp",
                side_effect=latest_timestamp_side_effect,
            )
        else:
            timestamp_patch = patch(
                "src.prediction.db.load_latest_prediction_timestamp",
                return_value="2024-01-01T00:00:00",
            )

        if evaluate_side_effect is not None:
            # [2.1/5] の本体 evaluate_output_invariants 自体を例外で落とし、
            # A-0（評価未実行）経路を再現する。
            evaluate_patch = patch(
                "src.prediction.output_invariants.evaluate_output_invariants",
                side_effect=evaluate_side_effect,
            )
        else:
            # 通常ケースは実関数のまま評価させたいので何も patch しない。
            evaluate_patch = nullcontext()

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
            timestamp_patch,
            patch(
                "src.prediction.db.prediction_results.load_run_stats_at",
                return_value=previous_raw,
            ),
            evaluate_patch,
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
            previous_raw=None,
        )
        self.assertIsNotNone(results)
        rule = next(r for r in results if r.rule_id == "NF-303-5")
        self.assertTrue(rule.triggered)
        self.assertIn("A-2", rule.details["violation_ids"])
        # I-1: notifier が send_webhook_notification そのものであることを検証。
        # ここが欠落すると run_conditional_notification 内で notifier=None と
        # なり、評価は正しくても Discord には永遠に届かない（完全サイレント劣化）。
        self.assertIs(self._captured["notifier"], send_webhook_notification)

    def test_healthy_run_is_not_triggered(self):
        results = self._run_pipeline(
            output_rows=_rows(705, model_count=2),
            loaded_models=["UnifiedStockXGBoost", "UnifiedStockLightGBM"],
            previous_raw=None,
        )
        rule = next(r for r in results if r.rule_id == "NF-303-5")
        self.assertFalse(rule.triggered)
        self.assertIs(self._captured["notifier"], send_webhook_notification)

    def test_evaluation_exception_fires_a0_without_stopping_pipeline(self):
        """[2.1/5] が例外で中断しても NF-303-5 は A-0（評価未実行）として発火し、
        パイプライン自体は継続する（NON_CRITICAL）。

        C-1 の修正で「前回ラン統計の取得」は [2/5] 側の内側 try/except に
        移動し、失敗しても previous_stats=None に縮退するだけで A-0 には
        ならない（意図した挙動）。そのため A-0 を再現するには、[2.1/5] の
        本体である evaluate_output_invariants 自体を例外で落とす必要がある。
        prediction_violation_ids は [2/5] 後の初期値 None のまま [6/6] に渡り、
        check_prediction_output_rule(None) が violation_ids=["A-0"] / triggered=True
        を返す設計（#615 対策のフェイルセーフ）をここで固定する。
        """
        results = self._run_pipeline(
            output_rows=_rows(705, model_count=2),
            loaded_models=["UnifiedStockXGBoost", "UnifiedStockLightGBM"],
            previous_raw=None,
            evaluate_side_effect=RuntimeError("evaluation exploded"),
        )
        self.assertIsNotNone(results)
        rule = next(r for r in results if r.rule_id == "NF-303-5")
        self.assertTrue(rule.triggered)
        self.assertIn("A-0", rule.details["violation_ids"])

    def test_previous_stats_fetch_failure_degrades_without_a0(self):
        """[2/5] 内の前回ラン統計取得が例外で失敗しても、
        previous_stats=None に縮退するだけでパイプラインは継続し、
        A-0（評価未実行）にはならない（[2.1/5] 自体は正常に走るため）。

        load_latest_prediction_timestamp の例外注入で再現する。
        """
        results = self._run_pipeline(
            output_rows=_rows(705, model_count=2),
            loaded_models=["UnifiedStockXGBoost", "UnifiedStockLightGBM"],
            previous_raw=None,
            latest_timestamp_side_effect=RuntimeError("db unavailable"),
        )
        self.assertIsNotNone(results)
        rule = next(r for r in results if r.rule_id == "NF-303-5")
        # 前回統計が取れないだけで、絶対値チェック（A系）は正常なので発火しない。
        self.assertFalse(rule.triggered)
        self.assertNotIn("A-0", rule.details.get("violation_ids", []))
        self.assertFalse(rule.details.get("compared_with_previous", True))

    def test_previous_run_present_reaches_notification_with_b_violation(self):
        """前回ラン統計が存在するケースで B 系の急変チェックが実際に走り、
        違反が violation_ids に載って通知まで届くこと。

        C-1 修正前は保存後に前回ランを探していたため、Delete-Insert方式の
        prediction_results では今回の保存で前回ランの行が上書き済みとなり、
        previous_stats は常に None（急変チェックは永遠にスキップ）だった。
        この経路が end-to-end で一度も踏まれていなかったのが C-1 の欠陥。

        前回ラン: 705銘柄, model_count=2 で固定（急変なし）、diff_ratio は
        ばらつきを持たせて stdev を安定させる。
        今回ラン: 100銘柄まで急減させ B-1（銘柄数急減）を発火させる。
        model_count は前回と同じ 2 のまま（A-1/A-2 は発火させない）。
        """
        previous_model_counts = [2] * 705
        previous_diff_ratios = [0.001 * (i - 352) for i in range(705)]

        current_rows = _rows(100, model_count=2, diff_ratio_scale=0.001)

        results = self._run_pipeline(
            output_rows=current_rows,
            loaded_models=["UnifiedStockXGBoost", "UnifiedStockLightGBM"],
            previous_raw=(previous_model_counts, previous_diff_ratios),
        )

        self.assertIsNotNone(results)
        rule = next(r for r in results if r.rule_id == "NF-303-5")
        self.assertTrue(rule.details.get("compared_with_previous"))
        # 705→100 の急減幅に対し diff_ratio のレンジも大きく縮めているため
        # B-3（分散急縮小）も同時に発火しうる。B-1 が「その他の副作用で偶然
        # 通っただけ」にならないよう、B-1 そのものの存在を件数で固定する。
        self.assertEqual(rule.details["violation_ids"].count("B-1"), 1)
        b1 = next(v for v in rule.details["violations"] if v["id"] == "B-1")
        self.assertIn("705", b1["description"])
        self.assertIn("100", b1["description"])
        self.assertTrue(rule.triggered)
        self.assertIs(self._captured["notifier"], send_webhook_notification)

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
                previous_raw=None,
                predict_side_effect=RuntimeError("boom"),
            )


if __name__ == "__main__":
    unittest.main()
