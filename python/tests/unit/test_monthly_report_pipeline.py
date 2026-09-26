"""ユニットテスト: monthly_report_pipeline (R-203)"""

from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.infrastructure.in_memory import InMemoryAnalyticsQuery
from src.reporting.kpi import (
    _EMPTY_DIFF,
    _REPORT_DAYS,
    MonthlyKPI,
    _compute_drift_count,
    _compute_hit_rate,
    get_monthly_kpis,
)
from src.reporting.monthly import (
    _load_latest_wf_summary,
    _load_monthly_kpis,
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
            "src.reporting.monthly.get_results_dir",
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
                "src.reporting.monthly.get_results_dir",
                return_value=tmpdir,
            ):
                df, filename = _load_latest_wf_summary()

        self.assertIsNotNone(df)
        self.assertEqual(filename, "wf_summary_20260401_120000.csv")
        self.assertEqual(len(df), 1)


# ---------------------------------------------------------------------------
# _compute_hit_rate (kpi_service)
# ---------------------------------------------------------------------------


class TestComputeHitRate(unittest.TestCase):
    def test_returns_none_when_table_empty(self):
        result = _compute_hit_rate(pd.DataFrame())
        self.assertIsNone(result)

    def test_calculates_mean_direction_match(self):
        df = pd.DataFrame(
            {
                "direction_match": [True, True, False, True],
                # checked_at を含まない → 日付フィルタなしでそのまま集計
            }
        )
        result = _compute_hit_rate(df)
        self.assertAlmostEqual(result, 0.75)

    def test_filters_by_checked_at(self):
        now = datetime.now()
        old_date = "2020-01-01"
        recent_date = now.strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "direction_match": [True, False],
                "checked_at": [recent_date, old_date],
            }
        )
        result = _compute_hit_rate(df, days=30)
        # old_dateは30日超のため除外 → recent_date の True のみ → 1.0
        self.assertAlmostEqual(result, 1.0)

    def test_does_not_mutate_callers_dataframe(self):
        """checked_at の型変換は内部コピーに対して行い、渡された DataFrame を書き換えない。"""
        df = pd.DataFrame({"direction_match": [True], "checked_at": ["2020-01-01"]})
        _compute_hit_rate(df)
        self.assertEqual(df["checked_at"].iloc[0], "2020-01-01")


class TestComputeDriftCount(unittest.TestCase):
    def test_returns_zero_when_empty(self):
        self.assertEqual(_compute_drift_count(pd.DataFrame()), 0)

    def test_counts_rows_over_either_threshold(self):
        df = pd.DataFrame(
            {
                "mean_abs_error": [0.05, 0.01, 0.01],
                "direction_accuracy": [0.60, 0.40, 0.60],
            }
        )
        # 1 行目は誤差超過、2 行目は正解率割れ、3 行目は健全
        self.assertEqual(_compute_drift_count(df), 2)


# ---------------------------------------------------------------------------
# _compute_avg_slippage (kpi_service)
# ---------------------------------------------------------------------------


def _kpis(diff_summary):
    return get_monthly_kpis(
        diff_summary=diff_summary, accuracy_df=pd.DataFrame(), drift_df=pd.DataFrame()
    )


class TestComputeAvgSlippage(unittest.TestCase):
    def test_avg_slippage_comes_from_injected_summary(self):
        kpi = _kpis({"avg_paper_slippage": 0.0123})
        self.assertAlmostEqual(kpi.avg_slippage, 0.0123)

    def test_returns_none_when_key_missing(self):
        kpi = _kpis({})
        self.assertIsNone(kpi.avg_slippage)

    def test_none_diff_summary_means_avg_slippage_is_none_not_zero(self):
        """diff_summary=None（読み取り失敗を表す）のとき avg_slippage は None であり、
        _EMPTY_DIFF の 0.0 と混同してはならない（false な「計測されたゼロ」を防ぐ）。
        """
        kpi = _kpis(None)
        self.assertIsNone(kpi.avg_slippage)
        self.assertEqual(kpi.diff_summary, _EMPTY_DIFF)


# ---------------------------------------------------------------------------
# _load_monthly_kpis（入口がポートから材料を取得する）
# ---------------------------------------------------------------------------


class TestLoadMonthlyKpis(unittest.TestCase):
    def test_queries_port_with_monthly_window(self):
        """精度は horizon=1/limit=5000、ドリフトは recent_n=_REPORT_DAYS で問い合わせる。"""
        analytics = InMemoryAnalyticsQuery()
        _load_monthly_kpis(analytics)
        self.assertEqual(
            analytics.calls,
            [
                ("paper_real_diff_summary", {"recent_days": _REPORT_DAYS}),
                (
                    "prediction_accuracy",
                    {"market": None, "symbol": None, "horizon": 1, "limit": 5000},
                ),
                ("drift_summary", {"horizon": 1, "recent_n": _REPORT_DAYS}),
            ],
        )

    def test_aggregates_port_data(self):
        analytics = InMemoryAnalyticsQuery(
            accuracy=pd.DataFrame({"direction_match": [True, False, True, True]}),
            drift=pd.DataFrame({"mean_abs_error": [0.05, 0.01], "direction_accuracy": [0.6, 0.6]}),
        )
        kpi = _load_monthly_kpis(analytics)
        self.assertAlmostEqual(kpi.hit_rate, 0.75)
        self.assertEqual(kpi.drift_count, 1)

    def test_diff_summary_failure_is_reported_as_missing(self):
        class _FailingDiff(InMemoryAnalyticsQuery):
            def paper_real_diff_summary(self, recent_days: int = 7) -> dict:
                raise RuntimeError("DB down")

        kpi = _load_monthly_kpis(_FailingDiff())
        self.assertIsNone(kpi.avg_slippage)

    def test_accuracy_failure_propagates(self):
        """精度の取得失敗は握りつぶさず呼び出し元へ伝播させる（従来挙動）。"""

        class _FailingAccuracy(InMemoryAnalyticsQuery):
            def prediction_accuracy(self, market=None, symbol=None, horizon=1, limit=500):
                raise RuntimeError("DB down")

        with self.assertRaises(RuntimeError):
            _load_monthly_kpis(_FailingAccuracy())


# ---------------------------------------------------------------------------
# run_monthly_report
# ---------------------------------------------------------------------------


class TestRunMonthlyReport(unittest.TestCase):
    @patch("src.reporting.monthly.get_monthly_kpis")
    @patch(
        "src.reporting.monthly._load_latest_wf_summary",
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
    def test_aggregates_all_kpi_fields(self, _mock_wf, mock_kpi):
        mock_kpi.return_value = MonthlyKPI(hit_rate=0.6, avg_slippage=0.001, drift_count=0)
        summary = run_monthly_report(target_month="2026-04", analytics=InMemoryAnalyticsQuery())

        self.assertEqual(summary.target_month, "2026-04")
        self.assertAlmostEqual(summary.net_return, 0.05)
        self.assertAlmostEqual(summary.sharpe_ratio, 1.2)
        self.assertAlmostEqual(summary.max_drawdown, -0.09)
        self.assertAlmostEqual(summary.hit_rate, 0.6)
        self.assertAlmostEqual(summary.avg_slippage, 0.001)
        self.assertEqual(summary.symbol_count, 2)
        self.assertEqual(summary.wf_snapshot_file, "wf_summary_20260401.csv")

    @patch("src.reporting.monthly.get_monthly_kpis")
    @patch(
        "src.reporting.monthly._load_latest_wf_summary",
        return_value=(None, None),
    )
    def test_returns_none_kpis_when_no_data(self, _mock_wf, mock_kpi):
        mock_kpi.return_value = MonthlyKPI(hit_rate=None, avg_slippage=None, drift_count=0)
        summary = run_monthly_report(target_month="2026-04", analytics=InMemoryAnalyticsQuery())

        self.assertIsNone(summary.net_return)
        self.assertIsNone(summary.sharpe_ratio)
        self.assertIsNone(summary.max_drawdown)
        self.assertIsNone(summary.hit_rate)
        self.assertIsNone(summary.avg_slippage)
        self.assertIsNone(summary.symbol_count)
        self.assertIsNone(summary.wf_snapshot_file)

    @patch("src.reporting.monthly.get_monthly_kpis")
    @patch(
        "src.reporting.monthly._load_latest_wf_summary",
        return_value=(None, None),
    )
    def test_uses_current_month_when_not_specified(self, _mock_wf, mock_kpi):
        mock_kpi.return_value = MonthlyKPI(hit_rate=None, avg_slippage=None, drift_count=0)
        expected_month = datetime.now().strftime("%Y-%m")
        summary = run_monthly_report(analytics=InMemoryAnalyticsQuery())
        self.assertEqual(summary.target_month, expected_month)


if __name__ == "__main__":
    unittest.main()
