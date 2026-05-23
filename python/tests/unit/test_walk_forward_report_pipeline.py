"""Unit Test: walk_forward_report_pipeline"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.backtest.walk_forward_report import _build_comparison, _summarize_wf_result


def test_summarize_wf_result_returns_mean_values():
    wf_df = pd.DataFrame(
        {
            "total_return": [0.01, 0.03],
            "sharpe_ratio": [1.0, 1.4],
            "max_drawdown": [-0.10, -0.06],
            "cost_impact_return": [0.002, 0.003],
        }
    )

    summary = _summarize_wf_result(wf_df, market="jp", symbol="7203")

    assert summary["market"] == "jp"
    assert summary["symbol"] == "7203"
    assert summary["fold_count"] == 2
    assert summary["total_return"] == 0.02
    assert summary["sharpe_ratio"] == 1.2


def test_build_comparison_with_previous_snapshot_adds_delta_columns():
    current = pd.DataFrame(
        [
            {"market": "jp", "symbol": "7203", "total_return": 0.03, "sharpe_ratio": 1.3},
            {"market": "us", "symbol": "AAPL", "total_return": 0.02, "sharpe_ratio": 1.1},
        ]
    )
    prev = pd.DataFrame(
        [
            {"market": "jp", "symbol": "7203", "total_return": 0.01, "sharpe_ratio": 1.0},
            {"market": "us", "symbol": "AAPL", "total_return": 0.04, "sharpe_ratio": 1.2},
        ]
    )

    compared = _build_comparison(current, prev)

    row_jp = compared[(compared["market"] == "jp") & (compared["symbol"] == "7203")].iloc[0]
    row_us = compared[(compared["market"] == "us") & (compared["symbol"] == "AAPL")].iloc[0]

    assert "delta_total_return" in compared.columns
    assert row_jp["delta_total_return"] == pytest.approx(0.02)
    assert row_us["delta_total_return"] == pytest.approx(-0.02)
    assert bool(row_jp["has_previous"]) is True


def test_build_comparison_without_previous_marks_no_history():
    current = pd.DataFrame([{"market": "jp", "symbol": "7203", "total_return": 0.03}])

    compared = _build_comparison(current, prev_df=None)

    assert "has_previous" in compared.columns
    assert bool(compared.iloc[0]["has_previous"]) is False


# ---------------------------------------------------------------------------
# 追加テスト: _find_latest_summary_csv / _to_markdown_summary / _numeric_mean /
#             run_walk_forward_comparison_report
# ---------------------------------------------------------------------------


class TestFindLatestSummaryCsv:
    """_find_latest_summary_csv のテスト"""

    def test_returns_none_when_no_csv(self, tmp_path):
        """CSV が存在しない場合は None が返ること"""
        from src.backtest.walk_forward_report import _find_latest_summary_csv

        result = _find_latest_summary_csv(tmp_path)
        assert result is None

    def test_returns_latest_csv_when_multiple_exist(self, tmp_path):
        """複数 CSV がある場合は最新のものが返ること（名前の降順）"""
        from src.backtest.walk_forward_report import _find_latest_summary_csv

        older = tmp_path / "wf_summary_20260101_120000.csv"
        newer = tmp_path / "wf_summary_20260201_120000.csv"
        older.write_text("col\n1", encoding="utf-8")
        newer.write_text("col\n2", encoding="utf-8")

        result = _find_latest_summary_csv(tmp_path)

        assert result is not None
        assert result.name == newer.name

    def test_returns_none_when_only_excluded(self, tmp_path):
        """唯一の CSV が exclude に一致する場合は None が返ること"""
        from src.backtest.walk_forward_report import _find_latest_summary_csv

        csv = tmp_path / "wf_summary_20260101_120000.csv"
        csv.write_text("col\n1", encoding="utf-8")

        result = _find_latest_summary_csv(tmp_path, exclude=csv)
        assert result is None


class TestToMarkdownSummary:
    """_to_markdown_summary のテスト"""

    def test_returns_markdown_string(self):
        """Markdown 文字列が返ること"""
        from src.backtest.walk_forward_report import _to_markdown_summary

        df = pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "7203",
                    "total_return": 0.03,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": -0.08,
                    "has_previous": False,
                }
            ]
        )
        result = _to_markdown_summary(df, previous_path=None)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Walk-Forward" in result

    def test_handles_empty_dataframe(self):
        """空 DataFrame でも例外なく文字列が返ること"""
        from src.backtest.walk_forward_report import _to_markdown_summary

        result = _to_markdown_summary(pd.DataFrame(), previous_path=None)

        assert isinstance(result, str)


class TestNumericMean:
    """_numeric_mean のテスト"""

    def test_calculates_mean(self):
        """指定列の平均値が返ること"""
        from src.backtest.walk_forward_report import _numeric_mean

        df = pd.DataFrame({"total_return": [0.01, 0.03, 0.02]})
        result = _numeric_mean(df, ["total_return"])

        assert abs(result["total_return"] - 0.02) < 1e-6

    def test_returns_empty_for_missing_column(self):
        """列が存在しない場合は空辞書が返ること"""
        from src.backtest.walk_forward_report import _numeric_mean

        df = pd.DataFrame({"other": [1, 2]})
        result = _numeric_mean(df, ["nonexistent"])

        assert result == {}

    def test_calculates_mean_for_multiple_columns(self):
        """複数列の平均値が返ること"""
        from src.backtest.walk_forward_report import _numeric_mean

        df = pd.DataFrame({"total_return": [0.01, 0.03], "sharpe_ratio": [1.0, 3.0]})
        result = _numeric_mean(df, ["total_return", "sharpe_ratio"])

        assert abs(result["total_return"] - 0.02) < 1e-6
        assert abs(result["sharpe_ratio"] - 2.0) < 1e-6


class TestRunWalkForwardComparisonReport:
    """run_walk_forward_comparison_report のテスト"""

    def test_runs_for_each_symbol(self, tmp_path):
        """各銘柄でバックテストが実行され、結果辞書が返ること"""
        from src.backtest.walk_forward_report import run_walk_forward_comparison_report
        from src.domain.types import SymbolTask

        reports_dir = tmp_path / "backtest" / "walk_forward_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        wf_df = pd.DataFrame(
            {
                "total_return": [0.02],
                "sharpe_ratio": [1.1],
                "max_drawdown": [-0.05],
                "cost_impact_return": [0.001],
            }
        )

        with (
            patch("src.backtest.walk_forward_report.load_target_symbols") as mock_symbols,
            patch("src.backtest.walk_forward_report.run_backtest_walk_forward") as mock_wf,
            patch(
                "src.backtest.walk_forward_report.get_results_dir",
                return_value=str(tmp_path),
            ),
            patch(
                "src.backtest.walk_forward_report.ensure_dir",
                return_value=str(reports_dir),
            ),
        ):
            mock_symbols.return_value = [SymbolTask(market="jp", symbol="7203")]
            mock_wf.return_value = (None, None, wf_df)

            result = run_walk_forward_comparison_report(limit_symbols=1)

        mock_wf.assert_called_once()
        mock_symbols.assert_called_once_with(as_of_date=None)
        assert result["total"] == 1
        assert result["failed"] == 0

    def test_passes_as_of_date_to_symbol_loader(self, tmp_path):
        """as_of_date が load_target_symbols に渡されること"""
        from src.backtest.walk_forward_report import run_walk_forward_comparison_report
        from src.domain.types import SymbolTask

        reports_dir = tmp_path / "backtest" / "walk_forward_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        wf_df = pd.DataFrame(
            {"total_return": [0.01], "sharpe_ratio": [1.0], "max_drawdown": [-0.02]}
        )

        with (
            patch("src.backtest.walk_forward_report.load_target_symbols") as mock_symbols,
            patch("src.backtest.walk_forward_report.run_backtest_walk_forward") as mock_wf,
            patch(
                "src.backtest.walk_forward_report.get_results_dir",
                return_value=str(tmp_path),
            ),
            patch(
                "src.backtest.walk_forward_report.ensure_dir",
                return_value=str(reports_dir),
            ),
        ):
            mock_symbols.return_value = [SymbolTask(market="us", symbol="AAPL")]
            mock_wf.return_value = (None, None, wf_df)

            run_walk_forward_comparison_report(as_of_date="2025-06-30")

        mock_symbols.assert_called_once_with(as_of_date="2025-06-30")
