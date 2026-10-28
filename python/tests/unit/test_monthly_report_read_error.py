"""月次レポートが予想の精度・ドリフトの読み込みの失敗を「データなし」と分けることのテスト（#1306）。"""

from __future__ import annotations

import tempfile
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.infrastructure.persistence.analytics_query import PostgresAnalyticsQuery
from src.reporting.monthly import run_monthly_report, save_monthly_report_to_file

_ACC = "src.prediction.db.accuracy"
_EMPTY_DIFF = {
    "tracked_count": 0,
    "comparable_count": 0,
    "avg_paper_slippage": 0.0,
    "avg_real_slippage": 0.0,
    "avg_abs_price_diff": 0.0,
    "avg_abs_diff_ratio": 0.0,
    "max_abs_price_diff": 0.0,
}


def _run_and_save(tmpdir: str) -> str:
    analytics = PostgresAnalyticsQuery()
    with (
        patch.object(PostgresAnalyticsQuery, "paper_real_diff_summary", return_value=_EMPTY_DIFF),
        patch("src.reporting.monthly.get_results_dir", return_value=tmpdir),
        patch("src.reporting.monthly._load_latest_wf_summary", return_value=(None, None)),
        patch(f"{_ACC}._db_connection", side_effect=lambda: nullcontext(MagicMock())),
    ):
        summary = run_monthly_report(target_month="2026-10", analytics=analytics)
        return save_monthly_report_to_file(summary, analytics=analytics)


@pytest.mark.parametrize("failing", ["prediction_accuracy", "drift"])
def test_read_failure_raises_and_writes_no_report(failing):
    """精度かドリフトの読み込みが落ちた月は、N/A と 0 のレポートを書かずに例外を投げる。"""

    def _read_sql(query, con, params=None):
        is_drift = "WITH ranked" in query
        if (failing == "drift") == is_drift:
            raise RuntimeError("db down")
        return pd.DataFrame()

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch(f"{_ACC}.pd.read_sql", side_effect=_read_sql):
            with pytest.raises(RuntimeError, match="db down"):
                _run_and_save(tmpdir)
        assert not list((Path(tmpdir) / "monthly").glob("*.md"))


def test_no_rows_still_saves_na_and_zero():
    """読めて0行の月は今までどおり Hit Rate N/A・閾値超過銘柄数 0 で保存する。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch(f"{_ACC}.pd.read_sql", return_value=pd.DataFrame()):
            path = _run_and_save(tmpdir)
        report = Path(path).read_text(encoding="utf-8")
    assert "| Hit Rate | N/A |" in report
    assert "閾値超過銘柄数: 0" in report


def test_default_callers_still_get_empty_frames():
    """raise_on_error を渡さない呼び出し（週次レポートなど）は今までどおり空の表で返る。"""
    analytics = PostgresAnalyticsQuery()
    with (
        patch(f"{_ACC}._db_connection", side_effect=lambda: nullcontext(MagicMock())),
        patch(f"{_ACC}.pd.read_sql", side_effect=RuntimeError("db down")),
    ):
        assert analytics.prediction_accuracy(horizon=1).empty
        assert analytics.drift_summary(horizon=1).empty
