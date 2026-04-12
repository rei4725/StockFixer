"""ユニットテスト: monthly_report_pipeline (R-203)"""

from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.services.monthly_report_pipeline import (
    _compute_avg_slippage,
    _compute_hit_rate,
    _load_latest_wf_summary,
    _mean_metric,
    run_monthly_report,
)

# ---------------------------------------------------------------------------
# _mean_metric
# ---------------------------------------------------------------------------


class TestMeanMetric(unittest.TestCase):
    def test_returns_float_when_column_exists(self):
        df = pd.DataFrame({"total_return": [0.01, 0.03]})
        result = _mean_metric(df, "total_return")
        self.assertAlmostEqual(result, 0.02)

    def test_returns_none_when_column_missing(self):
        df = pd.DataFrame({"other_col": [1.0]})
        result = _mean_metric(df, "total_return")
        self.assertIsNone(result)

    def test_returns_none_when_all_nan(self):
        df = pd.DataFrame({"total_return": [float("nan"), float("nan")]})
        result = _mean_metric(df, "total_return")
        self.assertIsNone(result)

    def test_ignores_non_numeric_values(self):
        df = pd.DataFrame({"sharpe_ratio": ["N/A", "1.5", "0.5"]})
        result = _mean_metric(df, "sharpe_ratio")
        self.assertAlmostEqual(result, 1.0)


# ---------------------------------------------------------------------------
# _load_latest_wf_summary
# ---------------------------------------------------------------------------


class TestLoadLatestWfSummary(unittest.TestCase):
    def test_returns_none_when_no_csv_found(self):
        with patch(
            "src.services.monthly_report_pipeline.get_results_dir",
            return_value="/nonexistent/path",
        ):
            df, filename = _load_latest_wf_summary()
        self.assertIsNone(df)
        self.assertIsNone(filename)

    def test_returns_latest_csv(self, tmp_path=None):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir) / "backtest" / "walk_forward_reports"
            report_dir.mkdir(parents=True)
            csv_path = report_dir / "wf_summary_20260401_120000.csv"
            sample_df = pd.DataFrame(
                [
                    {
                        "market": "jp",
                        "symbol": "7203",
                        "total_return": 0.05,
                        "sharpe_ratio": 1.2,
                        "max_drawdown": -0.08,
                    }
                ]
            )
            sample_df.to_csv(csv_path, index=False)

            with patch(
                "src.services.monthly_report_pipeline.get_results_dir",
                return_value=tmpdir,
            ):
                df, filename = _load_latest_wf_summary()

        self.assertIsNotNone(df)
        self.assertEqual(filename, "wf_summary_20260401_120000.csv")
        self.assertEqual(len(df), 1)


# ---------------------------------------------------------------------------
# _compute_hit_rate
# ---------------------------------------------------------------------------


class TestComputeHitRate(unittest.TestCase):
    @patch("src.services.monthly_report_pipeline.load_prediction_accuracy")
    def test_returns_none_when_table_empty(self, mock_load):
        mock_load.return_value = pd.DataFrame()
        result = _compute_hit_rate()
        self.assertIsNone(result)

    @patch("src.services.monthly_report_pipeline.load_prediction_accuracy")
    def test_calculates_mean_direction_match(self, mock_load):
        mock_load.return_value = pd.DataFrame(
            {
                "direction_match": [True, True, False, True],
                # checked_at を含まない → 日付フィルタなしでそのまま集計
            }
        )
        result = _compute_hit_rate()
        self.assertAlmostEqual(result, 0.75)

    @patch("src.services.monthly_report_pipeline.load_prediction_accuracy")
    def test_filters_by_checked_at(self, mock_load):
        now = datetime.now()
        old_date = "2020-01-01"
        recent_date = now.strftime("%Y-%m-%d")
        mock_load.return_value = pd.DataFrame(
            {
                "direction_match": [True, False],
                "checked_at": [recent_date, old_date],
            }
        )
        result = _compute_hit_rate(days=30)
        # old_dateは30日超のため除外 → recent_date の True のみ → 1.0
        self.assertAlmostEqual(result, 1.0)


# ---------------------------------------------------------------------------
# _compute_avg_slippage
# ---------------------------------------------------------------------------


class TestComputeAvgSlippage(unittest.TestCase):
    @patch("src.services.monthly_report_pipeline.load_paper_real_diff_summary")
    def test_returns_slippage_from_summary(self, mock_summary):
        mock_summary.return_value = {"avg_paper_slippage": 0.002}
        result = _compute_avg_slippage()
        self.assertAlmostEqual(result, 0.002)

    @patch("src.services.monthly_report_pipeline.load_paper_real_diff_summary")
    def test_returns_none_when_key_missing(self, mock_summary):
        mock_summary.return_value = {}
        result = _compute_avg_slippage()
        self.assertIsNone(result)

    @patch(
        "src.services.monthly_report_pipeline.load_paper_real_diff_summary",
        side_effect=Exception("DB error"),
    )
    def test_returns_none_on_exception(self, _mock):
        result = _compute_avg_slippage()
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# run_monthly_report
# ---------------------------------------------------------------------------


class TestRunMonthlyReport(unittest.TestCase):
    @patch("src.services.monthly_report_pipeline._compute_avg_slippage", return_value=0.001)
    @patch("src.services.monthly_report_pipeline._compute_hit_rate", return_value=0.6)
    @patch(
        "src.services.monthly_report_pipeline._load_latest_wf_summary",
        return_value=(
            pd.DataFrame(
                [
                    {
                        "market": "jp",
                        "symbol": "7203",
                        "total_return": 0.04,
                        "sharpe_ratio": 1.1,
                        "max_drawdown": -0.10,
                    },
                    {
                        "market": "us",
                        "symbol": "AAPL",
                        "total_return": 0.06,
                        "sharpe_ratio": 1.3,
                        "max_drawdown": -0.08,
                    },
                ]
            ),
            "wf_summary_20260401.csv",
        ),
    )
    def test_aggregates_all_kpi_fields(self, _mock_wf, _mock_hit, _mock_slip):
        summary = run_monthly_report(target_month="2026-04")

        self.assertEqual(summary.target_month, "2026-04")
        self.assertAlmostEqual(summary.net_return, 0.05)
        self.assertAlmostEqual(summary.sharpe_ratio, 1.2)
        self.assertAlmostEqual(summary.max_drawdown, -0.09)
        self.assertAlmostEqual(summary.hit_rate, 0.6)
        self.assertAlmostEqual(summary.avg_slippage, 0.001)
        self.assertEqual(summary.symbol_count, 2)
        self.assertEqual(summary.wf_snapshot_file, "wf_summary_20260401.csv")

    @patch("src.services.monthly_report_pipeline._compute_avg_slippage", return_value=None)
    @patch("src.services.monthly_report_pipeline._compute_hit_rate", return_value=None)
    @patch(
        "src.services.monthly_report_pipeline._load_latest_wf_summary",
        return_value=(None, None),
    )
    def test_returns_none_kpis_when_no_data(self, _mock_wf, _mock_hit, _mock_slip):
        summary = run_monthly_report(target_month="2026-04")

        self.assertIsNone(summary.net_return)
        self.assertIsNone(summary.sharpe_ratio)
        self.assertIsNone(summary.max_drawdown)
        self.assertIsNone(summary.hit_rate)
        self.assertIsNone(summary.avg_slippage)
        self.assertIsNone(summary.symbol_count)
        self.assertIsNone(summary.wf_snapshot_file)

    @patch("src.services.monthly_report_pipeline._compute_avg_slippage", return_value=None)
    @patch("src.services.monthly_report_pipeline._compute_hit_rate", return_value=None)
    @patch(
        "src.services.monthly_report_pipeline._load_latest_wf_summary",
        return_value=(None, None),
    )
    def test_uses_current_month_when_not_specified(self, _mock_wf, _mock_hit, _mock_slip):
        expected_month = datetime.now().strftime("%Y-%m")
        summary = run_monthly_report()
        self.assertEqual(summary.target_month, expected_month)


if __name__ == "__main__":
    unittest.main()
