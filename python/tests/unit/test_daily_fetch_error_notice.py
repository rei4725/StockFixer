"""日次の道の [1/6] で取得のエラーが線の下でも知らせるテスト（#1256）"""

from contextlib import ExitStack
from unittest.mock import patch

import pytest

from src.domain.types import BatchFailure, BatchResult, SymbolTask

_DAILY = "src.orchestration.jobs.daily"
_DISCORD = "src.reporting.discord.discord_utils"


def _batch(n_success: int, n_error: int, n_skip: int) -> BatchResult:
    return BatchResult(
        succeeded=[object()] * n_success,
        failed=[
            BatchFailure(market="us", symbol=f"S{i:03d}", error="接続取得がタイムアウト")
            for i in range(n_error)
        ],
        skipped=[object()] * n_skip,
    )


def _run_daily_pipeline(batch_result: BatchResult, total: int, tasks=None):
    """日次の道を外の依存を差し替えて流し、差し替えた mock を返す。"""
    from src.orchestration.jobs.daily import run_daily_pipeline

    if tasks is None:
        tasks = [SymbolTask(market="us", symbol=f"T{i}") for i in range(total)]
    with ExitStack() as stack:

        def p(target, **kwargs):
            return stack.enter_context(patch(target, **kwargs))

        mocks = {
            "webhook": p(f"{_DISCORD}.send_webhook_text"),
            "completion": p(f"{_DISCORD}.send_daily_pipeline_completion"),
            "error": p(f"{_DISCORD}.send_daily_pipeline_error"),
            "accuracy": p("src.prediction.prediction_pipeline.run_accuracy_check"),
            "predict": p("src.prediction.prediction_pipeline.predict_all_unified", return_value=[]),
        }
        p(f"{_DISCORD}.send_accuracy_summary")
        p("src.prediction.prediction_pipeline.output_top_worst_results")
        p("src.prediction.predict_unified.preload_models", return_value=[])
        p("src.prediction.shadow_evaluation.predict_with_challenger_unified", return_value=[])
        p("src.watchlist.batch_runner.load_target_symbols", return_value=tasks)
        p("src.market_data.pipeline.run_batch_pipeline", return_value=batch_result)
        p("src.infrastructure.discord_notification_adapter.DiscordNotificationAdapter")
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
        mocks["raised"] = None
        try:
            run_daily_pipeline()
        except RuntimeError as e:
            mocks["raised"] = e
    return mocks


class TestDailyFetchErrorNotice:
    def test_errors_below_line_send_one_notice_and_continue(self, caplog):
        """エラー 300 / 対象 784 は線の下でも1回知らせ、[2/6] 以降に進むこと"""
        with caplog.at_level("WARNING"):
            mocks = _run_daily_pipeline(_batch(483, 300, 1), total=784)

        assert mocks["raised"] is None
        mocks["webhook"].assert_called_once()
        text = mocks["webhook"].call_args.args[0]
        assert "[1/6] データ取得" in text
        assert "300" in text
        assert "784" in text
        assert "us_S000" in text
        assert "us_S299" not in text  # 先頭の数銘柄だけを並べる
        assert any("300" in r.getMessage() for r in caplog.records if r.levelname == "WARNING")
        mocks["accuracy"].assert_called()
        mocks["predict"].assert_called_once()
        mocks["completion"].assert_called_once()
        mocks["error"].assert_not_called()

    def test_skip_only_sends_no_notice(self):
        """エラー 0・スキップ 1 のふつうの朝は知らせないこと"""
        mocks = _run_daily_pipeline(_batch(783, 0, 1), total=784)

        assert mocks["raised"] is None
        mocks["webhook"].assert_not_called()
        mocks["error"].assert_not_called()
        mocks["predict"].assert_called_once()

    def test_errors_over_line_stop_without_double_notice(self):
        """線を超えた日は今までどおり CRITICAL で止まり、知らせを二重にしないこと"""
        mocks = _run_daily_pipeline(_batch(300, 483, 1), total=784)

        assert isinstance(mocks["raised"], RuntimeError)
        assert "多すぎる" in str(mocks["raised"])
        mocks["error"].assert_called_once()
        mocks["webhook"].assert_not_called()
        mocks["accuracy"].assert_not_called()
        mocks["predict"].assert_not_called()


@pytest.mark.parametrize("n_error", [6, 0])
def test_gate_notice_lists_head_symbols_with_market(n_error):
    """知らせの銘柄は市場つきで先頭の数銘柄に限ること"""
    from src.orchestration.jobs.unavailable_gates import _raise_if_fetch_mostly_unavailable

    with patch("src.orchestration.jobs.unavailable_gates._notify_stage_failure") as notify:
        _raise_if_fetch_mostly_unavailable(784, _batch(784 - n_error, n_error, 0))

    if n_error == 0:
        notify.assert_not_called()
        return
    notify.assert_called_once()
    message = str(notify.call_args.args[1])
    assert "us_S000" in message and "us_S004" in message
    assert "us_S005" not in message
    assert "ほか" in message
