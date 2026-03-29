"""Unit Test: walk_forward_report_pipeline"""

import pandas as pd
import pytest

from src.services.walk_forward_report_pipeline import _build_comparison, _summarize_wf_result


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
