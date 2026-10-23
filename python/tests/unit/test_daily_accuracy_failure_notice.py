"""日次の道の [2/6] 採点と採点の JSON が落ちた回の失敗の知らせのテスト（#1006）"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from src.orchestration.jobs.accuracy_json import write_accuracy_json_without_sectors

_DAILY = "src.orchestration.jobs.daily"
_DISCORD = "src.reporting.discord.discord_utils"


def _run_daily_pipeline(accuracy_side_effect, webhook_side_effect=None):
    """日次の道を外の依存を差し替えて流し、差し替えた mock を返す。"""
    from src.orchestration.jobs.daily import run_daily_pipeline

    with ExitStack() as stack:

        def p(target, **kwargs):
            return stack.enter_context(patch(target, **kwargs))

        mocks = {
            "webhook": p(f"{_DISCORD}.send_webhook_text", side_effect=webhook_side_effect),
            "completion": p(f"{_DISCORD}.send_daily_pipeline_completion"),
            "error": p(f"{_DISCORD}.send_daily_pipeline_error"),
            "accuracy_summary": p(f"{_DISCORD}.send_accuracy_summary"),
            "accuracy": p(
                "src.prediction.prediction_pipeline.run_accuracy_check",
                side_effect=accuracy_side_effect,
            ),
            "predict": p("src.prediction.prediction_pipeline.predict_all_unified", return_value=[]),
            "output": p("src.prediction.prediction_pipeline.output_top_worst_results"),
            "preload": p("src.prediction.predict_unified.preload_models", return_value=[]),
            "challenger": p(
                "src.prediction.shadow_evaluation.predict_with_challenger_unified",
                return_value=[],
            ),
        }
        p("src.watchlist.batch_runner.load_target_symbols", return_value=[])
        p("src.market_data.pipeline.run_batch_pipeline")
        p(f"{_DAILY}._raise_if_fetch_mostly_unavailable")
        p(f"{_DAILY}.production_already_predicted_today", return_value=False)
        p(f"{_DAILY}.write_accuracy_json_without_sectors", return_value=None)
        p(f"{_DAILY}.append_accuracy_json_sectors")
        p(f"{_DAILY}.fetch_previous_run_stats", return_value=None)
        p(f"{_DAILY}.evaluate_output_invariants_stage", return_value=([], {}))
        p(f"{_DAILY}.run_daily_drift_check")
        p(f"{_DAILY}.record_health_degraded_streak")
        p(f"{_DAILY}.evaluate_and_notify_alerts")
        p(
            "src.infrastructure.persistence.prediction_result_repository."
            "PostgresPredictionResultRepository"
        )

        run_daily_pipeline()
    return mocks


def _fail_production(*, model_name, **_kwargs):
    if model_name == "production":
        raise RuntimeError("db timeout")
    return None


class TestDailyAccuracyCheckFailureNotice:
    def test_production_failure_sends_notice_and_runs_later_stages(self):
        """production の採点が落ちた回は知らせを送り、[3/6] 以降を走らせること"""
        mocks = _run_daily_pipeline(_fail_production)

        mocks["webhook"].assert_called_once()
        text = mocks["webhook"].call_args.args[0]
        assert "[2/6] 予測精度チェック (production)" in text
        assert "db timeout" in text
        mocks["predict"].assert_called_once()
        mocks["output"].assert_called_once()
        mocks["completion"].assert_called_once()

    def test_success_sends_no_failure_notice(self):
        """採点が終わった回は失敗の知らせを送らないこと"""
        mocks = _run_daily_pipeline(None)

        mocks["webhook"].assert_not_called()
        mocks["error"].assert_not_called()
        mocks["predict"].assert_called_once()

    def test_notice_failure_does_not_stop_pipeline(self):
        """知らせ自体が落ちても [3/6] 以降を走らせること"""
        mocks = _run_daily_pipeline(_fail_production, RuntimeError("webhook down"))

        mocks["predict"].assert_called_once()
        mocks["completion"].assert_called_once()


class TestAccuracyJsonFailureNotice:
    @patch(f"{_DISCORD}.send_webhook_text")
    @patch(
        "src.prediction.accuracy_report.prepare_daily_accuracy_report",
        side_effect=RuntimeError("no rows"),
    )
    def test_prepare_failure_sends_notice(self, _mock_prepare, mock_webhook):
        """材料の作成が落ちた回は知らせを送り、None を返すこと"""
        assert write_accuracy_json_without_sectors() is None

        mock_webhook.assert_called_once()
        text = mock_webhook.call_args.args[0]
        assert "[2/6] 採点の JSON 書き出し" in text
        assert "no rows" in text

    @patch(f"{_DISCORD}.send_webhook_text")
    @patch("src.prediction.accuracy_report.prepare_daily_accuracy_report")
    def test_write_failure_sends_notice(self, mock_prepare, mock_webhook):
        """書き出しが落ちた回は知らせを送り、None を返すこと"""
        mock_prepare.return_value.write.side_effect = OSError("disk full")

        assert write_accuracy_json_without_sectors() is None

        mock_webhook.assert_called_once()
        assert "disk full" in mock_webhook.call_args.args[0]

    @patch(f"{_DISCORD}.send_webhook_text")
    @patch("src.prediction.accuracy_report.prepare_daily_accuracy_report")
    def test_success_sends_no_notice(self, mock_prepare, mock_webhook):
        """書き出しが終わった回は知らせを送らないこと"""
        mock_prepare.return_value = MagicMock()

        assert write_accuracy_json_without_sectors() is mock_prepare.return_value
        mock_webhook.assert_not_called()

    @patch(f"{_DISCORD}.send_webhook_text", side_effect=RuntimeError("webhook down"))
    @patch(
        "src.prediction.accuracy_report.prepare_daily_accuracy_report",
        side_effect=RuntimeError("no rows"),
    )
    def test_notice_failure_is_swallowed(self, _mock_prepare, _mock_webhook):
        """知らせ自体が落ちても例外を外へ出さず None を返すこと"""
        assert write_accuracy_json_without_sectors() is None
