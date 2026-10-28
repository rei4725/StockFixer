"""週次レポートが精度の読み込みの失敗を「データなし」と分けることのテスト（#1305）。"""

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pandas as pd

from src.infrastructure.persistence.analytics_query import PostgresAnalyticsQuery
from src.orchestration.jobs.weekly import run_weekly_report
from src.prediction.db import load_weekly_accuracy_summary

_ACC = "src.prediction.db.accuracy"
_DRIFT_DF = pd.DataFrame(
    [
        {
            "market": "jp",
            "symbol": "7203",
            "direction_accuracy": 0.6,
            "mean_abs_error": 0.01,
            "n_samples": 30,
        }
    ]
)


def _run(mock_notice: MagicMock, mock_report: MagicMock, mock_save: MagicMock) -> None:
    with patch.object(
        PostgresAnalyticsQuery, "paper_real_diff_summary", return_value={"tracked_count": 0}
    ), patch.object(
        PostgresAnalyticsQuery, "weekly_accuracy_snapshots", return_value=pd.DataFrame()
    ), patch(
        "src.prediction.db.save_weekly_accuracy_snapshot", mock_save
    ), patch(
        "src.reporting.discord.discord_utils.send_weekly_report", mock_report
    ), patch(
        "src.reporting.discord.discord_utils.send_webhook_text", mock_notice
    ), patch(
        "src.reporting.llm_review.generate_weekly_review", return_value=None
    ), patch(
        "src.infrastructure.llm.factory.get_text_review_port"
    ), patch(
        "src.prediction.db.load_top_prediction_misses", return_value=MagicMock()
    ), patch(
        "src.prediction.miss_analysis.run_miss_analysis_batch", return_value=[]
    ), patch(
        "src.reporting.discord.discord_utils.send_miss_analysis_summary"
    ), patch(
        "src.orchestration.jobs.weekly.run_weekly_paper_results"
    ):
        run_weekly_report()


@patch("src.prediction.db.load_drift_summary", return_value=_DRIFT_DF)
@patch(f"{_ACC}._db_connection", return_value=nullcontext(MagicMock()))
@patch(f"{_ACC}.pd.read_sql", side_effect=RuntimeError("could not write to disk"))
def test_weekly_summary_read_failure_sends_notice(mock_read_sql, mock_con, mock_drift):
    """週の記録の読み込みが落ちた週は、スナップショットを保存せず失敗を知らせる。"""
    mock_notice, mock_report, mock_save = MagicMock(), MagicMock(), MagicMock()
    _run(mock_notice, mock_report, mock_save)

    mock_save.assert_not_called()
    mock_report.assert_not_called()
    mock_notice.assert_called_once()
    text = mock_notice.call_args[0][0]
    assert "❌ 週次レポート生成失敗" in text
    assert "could not write to disk" in text


@patch("src.prediction.db.load_weekly_accuracy_summary", return_value=_DRIFT_DF)
@patch(f"{_ACC}._db_connection", return_value=nullcontext(MagicMock()))
@patch(f"{_ACC}.pd.read_sql", side_effect=RuntimeError("db down"))
def test_drift_read_failure_sends_notice(mock_read_sql, mock_con, mock_week):
    """drift の読み込みが落ちた週は「精度データなし」で黙らず失敗を知らせる。"""
    mock_notice, mock_report, mock_save = MagicMock(), MagicMock(), MagicMock()
    _run(mock_notice, mock_report, mock_save)

    mock_save.assert_not_called()
    mock_report.assert_not_called()
    mock_notice.assert_called_once()
    text = mock_notice.call_args[0][0]
    assert "❌ 週次レポート生成失敗" in text
    assert "db down" in text


@patch(f"{_ACC}._db_connection", side_effect=lambda: nullcontext(MagicMock()))
@patch(f"{_ACC}.pd.read_sql", return_value=pd.DataFrame())
def test_no_rows_week_does_not_notify(mock_read_sql, mock_con):
    """読めて0行の週は今までどおり例外にならず、レポートの段まで進む。"""
    mock_notice, mock_report, mock_save = MagicMock(), MagicMock(), MagicMock()
    _run(mock_notice, mock_report, mock_save)

    mock_notice.assert_not_called()
    mock_save.assert_called_once()
    assert mock_save.call_args.args[1].empty
    mock_report.assert_called_once()
    assert mock_report.call_args.kwargs["accuracy_df"].empty


@patch(f"{_ACC}._db_connection", return_value=nullcontext(MagicMock()))
@patch(f"{_ACC}.pd.read_sql", side_effect=RuntimeError("db down"))
def test_weekly_summary_default_returns_empty(mock_read_sql, mock_con):
    """既定の呼び方は今までどおり空の表で返る。"""
    assert load_weekly_accuracy_summary("2026-10-19").empty
