"""ユニットテスト: 月報の「市場別の的中率と基準」の節（#1816）"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

from src.infrastructure.in_memory import InMemoryAnalyticsQuery
from src.prediction.baseline_metrics import baseline_metrics
from src.reporting.monthly import save_monthly_report_to_file
from src.reporting.monthly_market_baseline import market_baseline_lines
from src.reporting.types import MonthlyReportSummary

_HEADER = "| 市場 | 件数 | 的中率 | 多数派の基準 | 差 | rank_corr |"


def _utc_stamp(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y%m%d_%H%M%S")


def _window() -> pd.DataFrame:
    """jp は4件中3件が当たり（上がる3・下がる1）、us は4件中1件が当たり（上がる3・下がる1）。

    窓の外の行と challenger の行は数えない。
    """
    recent = _utc_stamp(2)
    rows = [
        # jp: 予想 + + + -、実際 + + - -
        ("jp", "production", recent, 0.01, 0.02),
        ("jp", "production", recent, 0.02, 0.01),
        ("jp", "production", recent, 0.03, -0.01),
        ("jp", "production", recent, -0.01, -0.02),
        # us: 予想 - - + +、実際 + + + -
        ("us", "production", recent, -0.01, 0.02),
        ("us", "production", recent, -0.02, 0.01),
        ("us", "production", recent, 0.01, 0.03),
        ("us", "production", recent, 0.02, -0.01),
        # 数えない行
        ("us", "challenger", recent, 0.05, 0.05),
        ("jp", "production", _utc_stamp(60), 0.05, 0.05),
    ]
    return pd.DataFrame(
        rows,
        columns=["market", "model_name", "predicted_at", "predicted_ratio", "actual_ratio"],
    )


class TestMarketBaselineLines:
    def test_rows_per_market(self):
        lines = market_baseline_lines(_window(), baseline_metrics)
        assert lines[0] == "## 市場別の的中率と基準（直近30日）"
        assert _HEADER in lines
        # jp: 的中 3/4、上がった割合 2/4 → 多数派 50%、差 +25pt、rank_corr 0.20
        assert "| jp | 4 | 75.00% | 50.00% | +25.00pt | 0.20 |" in lines
        # us: 的中 1/4、上がった割合 3/4 → 多数派 75%、差 -50pt、rank_corr -0.20
        assert "| us | 4 | 25.00% | 75.00% | -50.00pt | -0.20 |" in lines
        assert lines.index("| jp | 4 | 75.00% | 50.00% | +25.00pt | 0.20 |") < lines.index(
            "| us | 4 | 25.00% | 75.00% | -50.00pt | -0.20 |"
        )

    def test_no_rows(self):
        lines = market_baseline_lines(pd.DataFrame(), baseline_metrics)
        assert lines[0] == "## 市場別の的中率と基準（直近30日）"
        assert "採点データなし" in lines
        assert _HEADER not in lines

    def test_no_market_column(self):
        df = _window().drop(columns=["market"])
        lines = market_baseline_lines(df, baseline_metrics)
        assert "採点データなし" in lines

    def test_no_baseline_fn(self):
        lines = market_baseline_lines(_window(), None)
        assert "基準の物差しが渡されていないため出さない" in lines
        assert _HEADER not in lines

    def test_rank_corr_none_shown_as_na(self):
        recent = _utc_stamp(1)
        df = pd.DataFrame(
            {
                "market": ["jp"],
                "model_name": ["production"],
                "predicted_at": [recent],
                "predicted_ratio": [0.01],
                "actual_ratio": [0.02],
            }
        )
        lines = market_baseline_lines(df, baseline_metrics)
        assert "| jp | 1 | 100.00% | 100.00% | +0.00pt | N/A |" in lines


class TestMonthlyReportHasSection:
    def test_section_written_between_kpi_and_weekly(self, tmp_path):
        analytics = InMemoryAnalyticsQuery(accuracy=_window())
        summary = MonthlyReportSummary(
            generated_at="2026-11-01T00:00:00",
            target_month="2026-10",
            net_return=None,
            max_drawdown=None,
            sharpe_ratio=None,
            hit_rate=None,
            avg_slippage=None,
            wf_snapshot_file=None,
            symbol_count=None,
        )
        with patch("src.reporting.monthly.get_results_dir", return_value=str(tmp_path)):
            path = save_monthly_report_to_file(
                summary, analytics=analytics, baseline_fn=baseline_metrics
            )
        text = open(path, encoding="utf-8").read()
        assert "| us | 4 | 25.00% | 75.00% | -50.00pt | -0.20 |" in text
        assert text.index("## KPI サマリー") < text.index("## 市場別の的中率と基準")
        assert text.index("## 市場別の的中率と基準") < text.index("## 週次 Hit Rate トレンド")
