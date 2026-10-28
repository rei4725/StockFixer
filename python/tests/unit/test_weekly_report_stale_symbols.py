"""週次レポートが、報告の週より後に予想していない銘柄を外すことのテスト（#1350）。"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.infrastructure.persistence.analytics_query import PostgresAnalyticsQuery
from src.orchestration.jobs.weekly import run_weekly_report

# 2026-10-29（木）JST に流すと、報告の週は 2026-10-19（月）から
_NOW_JST = datetime(2026, 10, 29, 9, 0, 0)


def _row(symbol: str, acc: float, last_at=None) -> dict:
    row = {
        "market": "us",
        "symbol": symbol,
        "direction_accuracy": acc,
        "mean_abs_error": 0.01,
        "n_samples": 30,
    }
    if last_at is not None:
        row["last_predicted_at"] = last_at
    return row


def _run(drift_df: pd.DataFrame) -> tuple[MagicMock, MagicMock]:
    mock_report, mock_review = MagicMock(), MagicMock(return_value=None)
    with patch("src.prediction.db.load_drift_summary", return_value=drift_df), patch(
        "src.prediction.db.load_weekly_accuracy_summary", return_value=pd.DataFrame()
    ), patch("src.prediction.db.save_weekly_accuracy_snapshot"), patch(
        "src.utils.japan_time.now_jst", return_value=_NOW_JST
    ), patch.object(
        PostgresAnalyticsQuery, "paper_real_diff_summary", return_value={"tracked_count": 0}
    ), patch.object(
        PostgresAnalyticsQuery, "weekly_accuracy_snapshots", return_value=pd.DataFrame()
    ), patch(
        "src.reporting.discord.discord_utils.send_weekly_report", mock_report
    ), patch(
        "src.reporting.discord.discord_utils.send_webhook_text"
    ), patch(
        "src.reporting.llm_review.generate_weekly_review", mock_review
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
    return mock_report, mock_review


def test_stale_symbol_is_dropped_from_report_and_review():
    """報告の週より前が最後の予想の銘柄は、レポートにも講評にも渡らない。"""
    drift_df = pd.DataFrame(
        [
            _row("AAPL", 0.6, "20261023_220000"),
            _row("MSFT", 0.4, "20261019_000000"),
            _row("WBD", 0.1, "20261015_220000"),
        ]
    )
    mock_report, mock_review = _run(drift_df)

    sent = mock_report.call_args.kwargs["accuracy_df"]
    reviewed = mock_review.call_args.args[0]
    assert set(sent["symbol"]) == {"AAPL", "MSFT"}
    assert set(reviewed["symbol"]) == {"AAPL", "MSFT"}
    assert sent["direction_accuracy"].mean() == pytest.approx(0.5)


def test_without_column_keeps_all_rows():
    """last_predicted_at の列が無い表は今までどおりすべて数える。"""
    drift_df = pd.DataFrame([_row("AAPL", 0.6), _row("WBD", 0.1)])
    mock_report, mock_review = _run(drift_df)

    assert set(mock_report.call_args.kwargs["accuracy_df"]["symbol"]) == {"AAPL", "WBD"}
    assert set(mock_review.call_args.args[0]["symbol"]) == {"AAPL", "WBD"}


def test_unreadable_time_row_is_kept():
    """時刻が読めない行は外さない。"""
    drift_df = pd.DataFrame(
        [
            _row("AAPL", 0.6, "20261023_220000"),
            _row("XYZ", 0.3, "not-a-time"),
            _row("NUL", 0.2, None),
        ]
    )
    drift_df.loc[2, "last_predicted_at"] = None
    mock_report, _ = _run(drift_df)

    assert set(mock_report.call_args.kwargs["accuracy_df"]["symbol"]) == {"AAPL", "XYZ", "NUL"}
