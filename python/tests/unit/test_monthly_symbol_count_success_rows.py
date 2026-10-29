"""月次レポートの集計銘柄数が WF サマリーの失敗の行を数えないことのテスト（#1375）"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.infrastructure.in_memory import InMemoryAnalyticsQuery
from src.reporting.kpi import MonthlyKPI
from src.reporting.monthly import run_monthly_report


def _ok(market: str, symbol: str, ret: float, sharpe: float, mdd: float) -> dict:
    return {
        "market": market,
        "symbol": symbol,
        "total_return": ret,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
    }


_SUCCESS_ROWS = [
    _ok("jp", "7203", 0.03, 1.0, -0.10),
    _ok("us", "AAPL", 0.06, 1.3, -0.08),
    _ok("us", "MSFT", 0.09, 1.6, -0.06),
]


def _run(df: pd.DataFrame):
    with (
        patch(
            "src.reporting.monthly._load_latest_wf_summary",
            return_value=(df, "wf_summary_20261028_160000.csv"),
        ),
        patch(
            "src.reporting.monthly.get_monthly_kpis",
            return_value=MonthlyKPI(hit_rate=None, avg_slippage=None, drift_count=0),
        ),
    ):
        return run_monthly_report(target_month="2026-10", analytics=InMemoryAnalyticsQuery())


def test_error_only_rows_are_not_counted():
    error_rows = [
        # 記録の保存だけが落ちた、成功の行と同じ銘柄の失敗の行
        {"market": "us", "symbol": "AAPL", "error": "Binder Error: ..."},
        {"market": "us", "symbol": "VYLR", "error": "stock_featuresにデータがありません"},
    ]
    summary = _run(pd.DataFrame(_SUCCESS_ROWS + error_rows))

    assert summary.symbol_count == 3
    assert summary.net_return == pytest.approx(0.06)
    assert summary.sharpe_ratio == pytest.approx(1.3)
    assert summary.max_drawdown == pytest.approx(-0.08)


def test_without_error_column_counts_all_rows():
    df = pd.DataFrame(_SUCCESS_ROWS)
    assert "error" not in df.columns

    summary = _run(df)

    assert summary.symbol_count == len(df)
    assert summary.net_return == pytest.approx(0.06)
    assert summary.sharpe_ratio == pytest.approx(1.3)
    assert summary.max_drawdown == pytest.approx(-0.08)


def test_all_error_rows_give_no_count():
    df = pd.DataFrame([{"market": "us", "symbol": "VYLR", "error": "x"}])
    df["total_return"] = float("nan")

    summary = _run(df)

    assert summary.symbol_count is None
    assert summary.net_return is None
