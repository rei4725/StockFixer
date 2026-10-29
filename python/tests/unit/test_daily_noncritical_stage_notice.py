"""日次の道の続けてよい段が例外で落ちた日に、後の段を止めずに1回知らせることのテスト（#1368）。

[4/6] ドリフト検査・[6/6] 運用アラート評価・[3.5/6] の段全体は、ERROR のログだけだと
自動の再学習や NF-303 の警報が黙って1日欠けるため、[2/6] と同じく Discord に知らせる。
"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

_JOBS = "src.orchestration.jobs.daily"


def _run_pipeline(drift_error=None, alert_error=None, shadow_error=None):
    """外への呼び出しを止めて日次の道を1回呼び、後の段の mock と知らせの文を返す。"""
    from src.orchestration.scheduler import run_daily_pipeline

    with ExitStack() as stack:

        def p(target, **kwargs):
            return stack.enter_context(patch(target, **kwargs))

        p("src.watchlist.batch_runner.load_target_symbols", return_value=[])
        p("src.market_data.pipeline.run_batch_pipeline", return_value=MagicMock())
        p("src.infrastructure.discord_notification_adapter.DiscordNotificationAdapter")
        p(f"{_JOBS}._raise_if_fetch_mostly_unavailable")
        p(f"{_JOBS}.production_already_predicted_today", return_value=False)
        p("src.prediction.prediction_pipeline.run_accuracy_check")
        p("src.reporting.discord.discord_utils.send_accuracy_summary")
        p("src.reporting.discord.discord_utils.send_daily_pipeline_error")
        p(f"{_JOBS}.write_accuracy_json_without_sectors", return_value=None)
        p("src.prediction.predict_unified.preload_models", return_value=[])
        p(f"{_JOBS}.fetch_previous_run_stats", return_value=None)
        p("src.prediction.prediction_pipeline.predict_all_unified", return_value=[])
        p("src.prediction.prediction_pipeline.output_top_worst_results")
        p(f"{_JOBS}.notify_prediction_errors")
        p(f"{_JOBS}.evaluate_output_invariants_stage", return_value=([], {}))
        p(
            "src.prediction.shadow_evaluation.predict_with_challenger_unified",
            return_value=[],
            side_effect=shadow_error,
        )
        p("src.prediction.shadow_evaluation.available_unified_challengers", return_value=[])
        drift = p(f"{_JOBS}.run_daily_drift_check", side_effect=drift_error)
        p(
            "src.infrastructure.persistence.prediction_result_repository."
            "PostgresPredictionResultRepository"
        )
        completion = p("src.reporting.discord.discord_utils.send_daily_pipeline_completion")
        p(f"{_JOBS}.record_health_degraded_streak")
        alerts = p(f"{_JOBS}.evaluate_and_notify_alerts", side_effect=alert_error)
        sectors = p(f"{_JOBS}.append_accuracy_json_sectors")
        webhook = p("src.reporting.discord.discord_utils.send_webhook_text")

        run_daily_pipeline()

    texts = [c.args[0] for c in webhook.call_args_list]
    return drift, completion, alerts, sectors, texts


def _notices(texts: list[str], label: str) -> list[str]:
    return [t for t in texts if t.startswith(f"❌ {label}")]


def test_drift_check_failure_notifies_once_and_continues():
    drift, completion, alerts, sectors, texts = _run_pipeline(
        drift_error=RuntimeError("精度の読み込み失敗")
    )

    assert len(_notices(texts, "[4/6] 日次ドリフトチェック")) == 1
    assert "精度の読み込み失敗" in _notices(texts, "[4/6]")[0]
    completion.assert_called_once()
    alerts.assert_called_once()
    sectors.assert_called_once()


def test_alert_evaluation_failure_notifies_once_and_appends_sectors():
    _, _, _, sectors, texts = _run_pipeline(alert_error=RuntimeError("streak 読み込み失敗"))

    assert len(_notices(texts, "[6/6] 運用アラート評価")) == 1
    sectors.assert_called_once()


def test_challenger_shadow_stage_failure_notifies_once_and_continues():
    drift, completion, alerts, _, texts = _run_pipeline(
        shadow_error=RuntimeError("challenger 読み込み失敗")
    )

    assert len(_notices(texts, "[3.5/6] Challenger shadow 予測")) == 1
    drift.assert_called_once()
    completion.assert_called_once()
    alerts.assert_called_once()


def test_no_notice_when_stages_succeed():
    drift, completion, alerts, sectors, texts = _run_pipeline()

    assert texts == []
    drift.assert_called_once()
    alerts.assert_called_once()
    sectors.assert_called_once()
