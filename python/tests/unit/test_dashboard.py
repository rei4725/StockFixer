"""ユニットテスト: dashboard (R-303)"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.reporting.dashboard import (
    _section_drift,
    _section_model_accuracy,
    _section_monthly_kpi,
    _section_paper_real_diff,
    run_dashboard,
)
from src.reporting.types import MonthlyReportSummary

# ---------------------------------------------------------------------------
# _section_monthly_kpi
# ---------------------------------------------------------------------------


class TestSectionMonthlyKpi(unittest.TestCase):
    @patch("src.reporting.dashboard.run_monthly_report")
    def test_returns_rows_with_all_kpis(self, mock_report):
        mock_report.return_value = MonthlyReportSummary(
            generated_at="2026-05-01T00:00:00",
            target_month="2026-05",
            net_return=0.05,
            max_drawdown=-0.10,
            sharpe_ratio=1.2,
            hit_rate=0.65,
            avg_slippage=0.001,
            symbol_count=10,
            wf_snapshot_file="wf_summary.csv",
        )
        rows = _section_monthly_kpi()
        self.assertIsNotNone(rows)
        labels = [r[0] for r in rows]
        self.assertIn("Net Return", labels)
        self.assertIn("Sharpe Ratio", labels)
        self.assertIn("Hit Rate", labels)

    @patch("src.reporting.dashboard.run_monthly_report")
    def test_handles_none_kpis(self, mock_report):
        mock_report.return_value = MonthlyReportSummary(
            generated_at="2026-05-01T00:00:00",
            target_month="2026-05",
            net_return=None,
            max_drawdown=None,
            sharpe_ratio=None,
            hit_rate=None,
            avg_slippage=None,
        )
        rows = _section_monthly_kpi()
        # N/A が表示されること
        values = [r[1] for r in rows]
        self.assertIn("N/A", values)


# ---------------------------------------------------------------------------
# _section_drift
# ---------------------------------------------------------------------------


class TestSectionDrift(unittest.TestCase):
    @patch("src.reporting.dashboard.load_drift_summary")
    def test_returns_empty_when_no_data(self, mock_load):
        mock_load.return_value = pd.DataFrame()
        rows, count = _section_drift(20)
        self.assertEqual(rows, [])
        self.assertEqual(count, 0)

    @patch("src.reporting.dashboard.load_drift_summary")
    def test_returns_exceeded_symbols(self, mock_load):
        mock_load.return_value = pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "7203",
                    "mean_abs_error": 0.05,
                    "direction_accuracy": 0.40,
                    "n_samples": 20,
                },
                {
                    "market": "jp",
                    "symbol": "9984",
                    "mean_abs_error": 0.01,
                    "direction_accuracy": 0.60,
                    "n_samples": 20,
                },
            ]
        )
        rows, count = _section_drift(20)
        # 7203 のみ閾値超過（mean_abs_error>=0.02 かつ direction_accuracy<=0.45）
        self.assertEqual(count, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "7203")

    @patch("src.reporting.dashboard.load_drift_summary")
    def test_no_exceeded_symbols(self, mock_load):
        mock_load.return_value = pd.DataFrame(
            [
                {
                    "market": "us",
                    "symbol": "AAPL",
                    "mean_abs_error": 0.005,
                    "direction_accuracy": 0.70,
                    "n_samples": 20,
                }
            ]
        )
        rows, count = _section_drift(20)
        self.assertEqual(count, 0)
        self.assertEqual(rows, [])


# ---------------------------------------------------------------------------
# _section_paper_real_diff
# ---------------------------------------------------------------------------


class TestSectionPaperRealDiff(unittest.TestCase):
    @patch("src.reporting.dashboard.load_paper_real_diff_summary")
    def test_returns_rows(self, mock_load):
        mock_load.return_value = {
            "tracked_count": 10,
            "comparable_count": 8,
            "avg_paper_slippage": 0.002,
            "avg_real_slippage": 0.003,
            "avg_abs_diff_ratio": 0.001,
            "max_abs_price_diff": 50.0,
        }
        rows = _section_paper_real_diff(30)
        labels = [r[0] for r in rows]
        self.assertIn("追跡件数", labels)
        self.assertIn("比較可能件数", labels)
        self.assertIn("Paper Slippage 平均", labels)


# ---------------------------------------------------------------------------
# _section_model_accuracy
# ---------------------------------------------------------------------------


class TestSectionModelAccuracy(unittest.TestCase):
    @patch("src.reporting.dashboard.load_experiment_runs")
    def test_returns_empty_when_no_data(self, mock_load):
        mock_load.return_value = pd.DataFrame()
        rows = _section_model_accuracy(100)
        self.assertEqual(rows, [])

    @patch("src.reporting.dashboard.load_experiment_runs")
    def test_aggregates_by_model_name(self, mock_load):
        mock_load.return_value = pd.DataFrame(
            [
                {
                    "model_name": "XGBoost",
                    "directional_accuracy": 0.60,
                    "rmse": 0.01,
                    "trained_at": "20260501_120000",
                },
                {
                    "model_name": "XGBoost",
                    "directional_accuracy": 0.70,
                    "rmse": 0.02,
                    "trained_at": "20260502_120000",
                },
                {
                    "model_name": "LightGBM",
                    "directional_accuracy": 0.55,
                    "rmse": 0.015,
                    "trained_at": "20260501_120000",
                },
            ]
        )
        rows = _section_model_accuracy(100)
        self.assertEqual(len(rows), 2)
        model_names = [r[0] for r in rows]
        self.assertIn("XGBoost", model_names)
        self.assertIn("LightGBM", model_names)


# ---------------------------------------------------------------------------
# run_dashboard（統合: 各セクションが例外でも継続する）
# ---------------------------------------------------------------------------


class TestRunDashboard(unittest.TestCase):
    @patch("src.reporting.dashboard._section_model_accuracy", return_value=[])
    @patch(
        "src.reporting.dashboard._section_paper_balance",
        return_value=[["残高 (現金)", "¥1,000,000"]],
    )
    @patch(
        "src.reporting.dashboard._section_paper_real_diff",
        return_value=[["追跡件数", "0"]],
    )
    @patch("src.reporting.dashboard._section_drift", return_value=([], 0))
    @patch(
        "src.reporting.dashboard._section_monthly_kpi",
        return_value=[["Net Return", "N/A"]],
    )
    def test_runs_without_error(self, *_mocks):
        """全モックで run_dashboard が例外なく完走することを確認する。"""
        run_dashboard(recent_days=7, drift_n=10)

    @patch(
        "src.reporting.dashboard._section_model_accuracy",
        side_effect=Exception("DB error"),
    )
    @patch(
        "src.reporting.dashboard._section_paper_balance",
        side_effect=Exception("DB error"),
    )
    @patch(
        "src.reporting.dashboard._section_paper_real_diff",
        side_effect=Exception("DB error"),
    )
    @patch(
        "src.reporting.dashboard._section_drift",
        side_effect=Exception("DB error"),
    )
    @patch(
        "src.reporting.dashboard._section_monthly_kpi",
        side_effect=Exception("DB error"),
    )
    def test_continues_on_section_failure(self, *_mocks):
        """各セクションが例外を起こしても run_dashboard が最後まで実行されること。"""
        # 例外が伝播しないことを確認（print で失敗メッセージが出るだけ）
        run_dashboard(recent_days=7, drift_n=10)


if __name__ == "__main__":
    unittest.main()
