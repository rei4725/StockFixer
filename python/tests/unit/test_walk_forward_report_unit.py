"""ユニットテスト: Walk-Forward 比較レポートの内部ヘルパー関数"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd


def _make_wf_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "total_return": [0.05, 0.06, 0.04],
            "sharpe_ratio": [0.5, 0.6, 0.4],
            "max_drawdown": [-0.10, -0.08, -0.12],
            "win_rate": [0.55, 0.60, 0.50],
            "num_trades": [10, 12, 8],
            "profit_factor": [1.5, 1.6, 1.4],
        }
    )


class TestNumericMean(unittest.TestCase):
    def test_returns_mean_of_specified_columns(self):
        from src.backtest.walk_forward_report import _numeric_mean

        df = _make_wf_df()
        result = _numeric_mean(df, ["total_return", "sharpe_ratio"])
        self.assertAlmostEqual(result["total_return"], 0.05, places=3)
        self.assertAlmostEqual(result["sharpe_ratio"], 0.5, places=3)

    def test_ignores_missing_columns(self):
        from src.backtest.walk_forward_report import _numeric_mean

        df = _make_wf_df()
        result = _numeric_mean(df, ["total_return", "nonexistent_col"])
        self.assertIn("total_return", result)
        self.assertNotIn("nonexistent_col", result)

    def test_empty_columns_returns_empty_dict(self):
        from src.backtest.walk_forward_report import _numeric_mean

        df = _make_wf_df()
        result = _numeric_mean(df, [])
        self.assertEqual(result, {})

    def test_all_columns_missing_returns_empty_dict(self):
        from src.backtest.walk_forward_report import _numeric_mean

        df = _make_wf_df()
        result = _numeric_mean(df, ["missing1", "missing2"])
        self.assertEqual(result, {})


class TestSummarizeWfResult(unittest.TestCase):
    def test_includes_market_and_symbol(self):
        from src.backtest.walk_forward_report import _summarize_wf_result

        df = _make_wf_df()
        result = _summarize_wf_result(df, "jp", "7203")
        self.assertEqual(result["market"], "jp")
        self.assertEqual(result["symbol"], "7203")

    def test_includes_fold_count(self):
        from src.backtest.walk_forward_report import _summarize_wf_result

        df = _make_wf_df()
        result = _summarize_wf_result(df, "jp", "7203")
        self.assertEqual(result["fold_count"], 3)

    def test_computes_numeric_mean(self):
        from src.backtest.walk_forward_report import _summarize_wf_result

        df = _make_wf_df()
        result = _summarize_wf_result(df, "jp", "7203")
        self.assertAlmostEqual(result["total_return"], 0.05, places=3)


class TestFindLatestSummaryCsv(unittest.TestCase):
    def test_returns_none_when_no_csvs(self):
        from src.backtest.walk_forward_report import _find_latest_summary_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            result = _find_latest_summary_csv(Path(tmpdir))
        self.assertIsNone(result)

    def test_returns_latest_csv(self):
        from src.backtest.walk_forward_report import _find_latest_summary_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "wf_summary_20240101_120000.csv"
            p2 = Path(tmpdir) / "wf_summary_20240102_120000.csv"
            p1.write_text("data")
            p2.write_text("data")
            result = _find_latest_summary_csv(Path(tmpdir))
        self.assertEqual(result.name, p2.name)

    def test_excludes_specified_path(self):
        from src.backtest.walk_forward_report import _find_latest_summary_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "wf_summary_20240101_120000.csv"
            p2 = Path(tmpdir) / "wf_summary_20240102_120000.csv"
            p1.write_text("data")
            p2.write_text("data")
            result = _find_latest_summary_csv(Path(tmpdir), exclude=p2)
        self.assertEqual(result.name, p1.name)


class TestBuildComparison(unittest.TestCase):
    def test_no_previous_sets_has_previous_false(self):
        from src.backtest.walk_forward_report import _build_comparison

        current = pd.DataFrame([{"market": "jp", "symbol": "7203", "total_return": 0.05}])
        result = _build_comparison(current, None)
        self.assertFalse(result["has_previous"].iloc[0])

    def test_empty_previous_sets_has_previous_false(self):
        from src.backtest.walk_forward_report import _build_comparison

        current = pd.DataFrame([{"market": "jp", "symbol": "7203", "total_return": 0.05}])
        result = _build_comparison(current, pd.DataFrame())
        self.assertFalse(result["has_previous"].iloc[0])

    def test_with_previous_computes_delta(self):
        from src.backtest.walk_forward_report import _build_comparison

        current = pd.DataFrame(
            [{"market": "jp", "symbol": "7203", "total_return": 0.06, "sharpe_ratio": 0.7}]
        )
        prev = pd.DataFrame(
            [{"market": "jp", "symbol": "7203", "total_return": 0.04, "sharpe_ratio": 0.5}]
        )
        result = _build_comparison(current, prev)
        self.assertIn("delta_total_return", result.columns)
        self.assertAlmostEqual(result["delta_total_return"].iloc[0], 0.02, places=5)

    def test_has_previous_true_when_prev_data_exists(self):
        from src.backtest.walk_forward_report import _build_comparison

        current = pd.DataFrame([{"market": "jp", "symbol": "7203", "total_return": 0.06}])
        prev = pd.DataFrame([{"market": "jp", "symbol": "7203", "total_return": 0.04}])
        result = _build_comparison(current, prev)
        self.assertTrue(result["has_previous"].iloc[0])


class TestToMarkdownSummary(unittest.TestCase):
    def test_returns_string(self):
        from src.backtest.walk_forward_report import _to_markdown_summary

        df = pd.DataFrame(
            [{"market": "jp", "symbol": "7203", "total_return": 0.05, "has_previous": False}]
        )
        result = _to_markdown_summary(df, None)
        self.assertIsInstance(result, str)
        self.assertIn("Walk-Forward", result)

    def test_empty_df_returns_data_not_found_message(self):
        from src.backtest.walk_forward_report import _to_markdown_summary

        result = _to_markdown_summary(pd.DataFrame(), None)
        self.assertIn("データがありません", result)

    def test_with_delta_and_has_previous_shows_top10_section(self):
        from src.backtest.walk_forward_report import _to_markdown_summary

        df = pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "7203",
                    "total_return": 0.06,
                    "total_return_current": 0.06,
                    "total_return_prev": 0.04,
                    "delta_total_return": 0.02,
                    "has_previous": True,
                }
            ]
        )
        result = _to_markdown_summary(df, None)
        self.assertIn("Total Return", result)

    def test_with_short_return_column_shows_short_section(self):
        from src.backtest.walk_forward_report import _to_markdown_summary

        df = pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "7203",
                    "total_return": 0.05,
                    "short_return": 0.01,
                    "has_previous": False,
                }
            ]
        )
        result = _to_markdown_summary(df, None)
        self.assertIn("ショート", result)


if __name__ == "__main__":
    unittest.main()
