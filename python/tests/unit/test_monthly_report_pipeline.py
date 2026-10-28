"""ユニットテスト: monthly_report_pipeline (R-203)"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
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
    save_monthly_report_to_file,
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

    def test_counts_only_production_rows(self):
        """challenger の影の予想の採点を production の的中率に混ぜない（#1090）。"""
        df = pd.DataFrame(
            {
                "direction_match": [True, False] * 5 + [True] * 8 + [False] * 2,
                "model_name": ["production"] * 10 + ["challenger"] * 10,
            }
        )
        kpi = get_monthly_kpis(diff_summary={}, accuracy_df=df, drift_df=pd.DataFrame())
        self.assertAlmostEqual(kpi.hit_rate, 0.50)

    def test_without_model_name_column_uses_all_rows(self):
        df = pd.DataFrame({"direction_match": [True, False] * 5 + [True] * 8 + [False] * 2})
        kpi = get_monthly_kpis(diff_summary={}, accuracy_df=df, drift_df=pd.DataFrame())
        self.assertAlmostEqual(kpi.hit_rate, 0.65)


class TestComputeDriftCount(unittest.TestCase):
    def test_returns_zero_when_empty(self):
        self.assertEqual(_compute_drift_count(pd.DataFrame()), 0)

    def test_counts_rows_over_either_threshold(self):
        df = pd.DataFrame(
            {
                "mean_abs_error": [0.05, 0.01, 0.01],
                "direction_accuracy": [0.60, 0.35, 0.60],
                "n_samples": [30, 30, 30],
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

    def test_zero_tracked_count_means_avg_slippage_is_none(self):
        """追跡件数が0なら、アダプタが補った 0.0 を計測値として扱わず None にする。"""
        kpi = _kpis(dict(_EMPTY_DIFF))
        self.assertIsNone(kpi.avg_slippage)


# ---------------------------------------------------------------------------
# _load_monthly_kpis（入口がポートから材料を取得する）
# ---------------------------------------------------------------------------


class _LimitedAccuracy(InMemoryAnalyticsQuery):
    """本物のアダプタと同じく predicted_at 降順の先頭 limit 行だけを返す。"""

    def prediction_accuracy(
        self, market=None, symbol=None, horizon=1, limit=500, raise_on_error=False
    ):
        df = super().prediction_accuracy(market=market, symbol=symbol, horizon=horizon, limit=limit)
        df = df.sort_values("predicted_at", ascending=False)
        return df.head(limit) if limit else df


class TestLoadMonthlyKpis(unittest.TestCase):
    def test_queries_port_with_monthly_window(self):
        """精度は horizon=1/limit=5000 から始め、ドリフトは recent_n=_REPORT_DAYS で問い合わせる。"""
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
            drift=pd.DataFrame(
                {
                    "mean_abs_error": [0.05, 0.01],
                    "direction_accuracy": [0.6, 0.6],
                    "n_samples": [30, 30],
                }
            ),
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

    def test_hit_rate_counts_all_rows_in_window_beyond_5000(self):
        """窓の中に 5000 行を超える採点があっても、窓の中の全行で方向一致率を出す。"""
        now = datetime.now()
        rows = []
        # 新しい側 5000 行はすべて的中、続く 3000 行（窓の中）はすべて外れ
        for i in range(8000):
            ts = now - timedelta(minutes=5 * i)
            rows.append(
                {
                    "predicted_at": ts.strftime("%Y%m%d_%H%M%S"),
                    "checked_at": ts,
                    "direction_match": i < 5000,
                }
            )
        # 窓より古い採点は数えない
        old = now - timedelta(days=_REPORT_DAYS + 10)
        rows.append(
            {
                "predicted_at": old.strftime("%Y%m%d_%H%M%S"),
                "checked_at": old,
                "direction_match": True,
            }
        )
        analytics = _LimitedAccuracy(accuracy=pd.DataFrame(rows))
        kpi = _load_monthly_kpis(analytics)
        self.assertAlmostEqual(kpi.hit_rate, 5000 / 8000)

    def test_accuracy_failure_propagates(self):
        """精度の取得失敗は握りつぶさず呼び出し元へ伝播させる（従来挙動）。"""

        class _FailingAccuracy(InMemoryAnalyticsQuery):
            def prediction_accuracy(
                self, market=None, symbol=None, horizon=1, limit=500, raise_on_error=False
            ):
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


# ---------------------------------------------------------------------------
# save_monthly_report_to_file（paper/real 乖離の表示）
# ---------------------------------------------------------------------------


def _render_report(diff_summary: dict) -> str:
    import tempfile

    analytics = InMemoryAnalyticsQuery(paper_real_diff=diff_summary)
    with tempfile.TemporaryDirectory() as tmpdir:
        with (
            patch("src.reporting.monthly.get_results_dir", return_value=tmpdir),
            patch("src.reporting.monthly._load_latest_wf_summary", return_value=(None, None)),
        ):
            summary = run_monthly_report(target_month="2026-10", analytics=analytics)
            path = save_monthly_report_to_file(summary, analytics=analytics)
        return Path(path).read_text(encoding="utf-8")


class TestSaveMonthlyReportPaperRealDiff(unittest.TestCase):
    def test_zero_comparable_count_shows_real_side_as_na(self):
        """比較可能件数が0なら、Real Slippage・乖離率・price 乖離を 0 と出さず N/A にする。"""
        diff = dict(_EMPTY_DIFF, tracked_count=7, avg_paper_slippage=0.0012)
        report = _render_report(diff)
        self.assertIn("| 比較可能件数 | 0 |", report)
        self.assertIn("| Real Slippage | N/A |", report)
        self.assertIn("| 平均乖離率 | N/A |", report)
        self.assertIn("| 平均price乖離 | N/A |", report)
        # 追跡件数はあるので Paper 側は計測値のまま
        self.assertIn("| Paper Slippage | 0.12% |", report)
        self.assertIn("| Avg Slippage | 0.12% |", report)

    def test_zero_tracked_count_shows_paper_slippage_as_na(self):
        """追跡件数が0なら、Paper Slippage と KPI の Avg Slippage も N/A にする。"""
        report = _render_report(dict(_EMPTY_DIFF))
        self.assertIn("| Paper Slippage | N/A |", report)
        self.assertIn("| Avg Slippage | N/A |", report)
        self.assertIn("| Real Slippage | N/A |", report)

    def test_counts_present_keep_measured_values(self):
        diff = {
            "tracked_count": 5,
            "comparable_count": 3,
            "avg_paper_slippage": 0.001,
            "avg_real_slippage": 0.002,
            "avg_abs_price_diff": 1.5,
            "avg_abs_diff_ratio": 0.0003,
            "max_abs_price_diff": 3.0,
        }
        report = _render_report(diff)
        self.assertIn("| 平均price乖離 | ¥1.50 |", report)
        self.assertIn("| 平均乖離率 | 0.0300% |", report)
        self.assertIn("| Paper Slippage | 0.10% |", report)
        self.assertIn("| Real Slippage | 0.20% |", report)
        self.assertIn("| Avg Slippage | 0.10% |", report)


if __name__ == "__main__":
    unittest.main()
