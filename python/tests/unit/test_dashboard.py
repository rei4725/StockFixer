"""ユニットテスト: dashboard (R-303)"""

from __future__ import annotations

import math
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.domain.ports import MarketDataPort
from src.infrastructure.in_memory import InMemoryAnalyticsQuery, InMemoryTradeDiffSink
from src.reporting.dashboard import (
    _section_drift,
    _section_model_accuracy,
    _section_monthly_kpi,
    _section_paper_balance,
    _section_paper_real_diff,
    run_dashboard,
)
from src.reporting.types import MonthlyReportSummary
from src.trading.brokers.paper import paper_broker
from src.trading.brokers.paper.paper_broker import PaperBroker
from src.utils.db._connection import _db_connection

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
        rows = _section_monthly_kpi(InMemoryAnalyticsQuery())
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
        rows = _section_monthly_kpi(InMemoryAnalyticsQuery())
        # N/A が表示されること
        values = [r[1] for r in rows]
        self.assertIn("N/A", values)


# ---------------------------------------------------------------------------
# _section_drift
# ---------------------------------------------------------------------------


class TestSectionDrift(unittest.TestCase):
    def test_returns_empty_when_no_data(self):
        analytics = InMemoryAnalyticsQuery(drift=pd.DataFrame())
        rows, count = _section_drift(analytics, 20)
        self.assertEqual(rows, [])
        self.assertEqual(count, 0)

    def test_queries_drift_with_given_window(self):
        """drift_n がそのまま recent_n として問い合わせに渡ること。"""
        analytics = InMemoryAnalyticsQuery()
        _section_drift(analytics, 20)
        self.assertEqual(analytics.calls, [("drift_summary", {"horizon": 1, "recent_n": 20})])

    def test_returns_exceeded_symbols(self):
        analytics = InMemoryAnalyticsQuery(
            drift=pd.DataFrame(
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
        )
        rows, count = _section_drift(analytics, 20)
        # 7203 のみ閾値超過（mean_abs_error>=0.02 かつ direction_accuracy<=0.45）
        self.assertEqual(count, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "7203")

    def test_no_exceeded_symbols(self):
        analytics = InMemoryAnalyticsQuery(
            drift=pd.DataFrame(
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
        )
        rows, count = _section_drift(analytics, 20)
        self.assertEqual(count, 0)
        self.assertEqual(rows, [])


# ---------------------------------------------------------------------------
# _section_paper_real_diff
# ---------------------------------------------------------------------------


class TestSectionPaperRealDiff(unittest.TestCase):
    def test_section_paper_real_diff_formats_percentages(self):
        rows = _section_paper_real_diff(
            {
                "tracked_count": 10,
                "comparable_count": 7,
                "avg_paper_slippage": 0.001,
                "avg_real_slippage": 0.002,
                "avg_abs_price_diff": 3.5,
                "avg_abs_diff_ratio": 0.0035,
                "max_abs_price_diff": 9.0,
            }
        )
        self.assertEqual(rows[0], ["追跡件数", "10"])
        self.assertEqual(rows[2], ["Paper Slippage 平均", "0.1000%"])


# ---------------------------------------------------------------------------
# _section_paper_balance（#1178）
# ---------------------------------------------------------------------------


def _paper_broker_at_close(monkeypatch, close: float) -> PaperBroker:
    """現在値（終値）が close の PaperBroker を返す。"""
    monkeypatch.setattr(paper_broker, "_today_jst", lambda: date(2026, 3, 15))
    market_data = MagicMock(spec=MarketDataPort)
    market_data.get_ohlcv.return_value = pd.DataFrame(
        {"Open": [close], "High": [close], "Low": [close], "Close": [close]},
        index=pd.to_datetime(["2026-03-15"]),
    )
    return PaperBroker(market_data_port=market_data, trade_diff_sink=InMemoryTradeDiffSink())


def _reset_paper_tables() -> None:
    with _db_connection() as con:
        con.execute("DELETE FROM paper_orders")
        con.execute("DELETE FROM paper_positions")
        con.execute("DELETE FROM paper_short_positions")
        con.execute("UPDATE paper_balance SET balance = 1000000.0")


def test_paper_balance_short_unrealized_from_valuation(monkeypatch):
    """空売りの含み損益が (avg_short_price - 現在値) × qty になり 0 にならない"""
    _reset_paper_tables()
    with _db_connection() as con:
        con.execute(
            "INSERT INTO paper_short_positions"
            " (symbol, qty, avg_short_price) VALUES ('7203', 100, 1000.0)"
        )
    broker = _paper_broker_at_close(monkeypatch, 1020.0)
    rows = dict(_section_paper_balance(broker))
    # (1,000 - 1,020) × 100 = -2,000
    assert rows["含み損益（空売り）"] == "¥-2,000"
    assert rows["空売りポジション数"] == "1"


def test_paper_balance_total_matches_pnl_summary(monkeypatch):
    """買い持ちと空売りの両方があるとき、合計損益が get_pnl_summary の total_pnl と一致する"""
    _reset_paper_tables()
    with _db_connection() as con:
        con.execute(
            "INSERT INTO paper_positions (symbol, qty, avg_price) VALUES ('6758', 100, 950.0)"
        )
        con.execute(
            "INSERT INTO paper_short_positions"
            " (symbol, qty, avg_short_price) VALUES ('7203', 100, 1100.0)"
        )
        con.execute("""
            INSERT INTO paper_orders
                (order_id, symbol, side, qty, price, order_type,
                 status, realized_pnl, filled_at)
            VALUES ('dash_pnl_01', '9984', 2, 100, 1000.0, 10,
                    'filled', 3000.0, CURRENT_TIMESTAMP)
            """)
    broker = _paper_broker_at_close(monkeypatch, 1000.0)
    rows = dict(_section_paper_balance(broker))
    total = broker.get_pnl_summary()["total_pnl"]
    # 実現 3,000 + 買い持ち (1,000 - 950) × 100 + 空売り (1,100 - 1,000) × 100 = 18,000
    assert total == pytest.approx(18000.0)
    assert rows["含み損益（買い持ち）"] == "¥5,000"
    assert rows["含み損益（空売り）"] == "¥10,000"
    assert rows["合計損益"] == f"¥{total:,.0f}"
    assert rows["評価できない建玉数"] == "0"


def test_paper_balance_counts_unvalued_positions():
    """評価できない（NaN の）建玉は 0 として黙って足さず、件数を行に出す"""
    broker = MagicMock()
    broker.get_pnl_summary.return_value = {"realized_pnl": 1000.0, "balance": 500000.0}
    broker.get_positions.return_value = [
        {"symbol": "6758", "qty": 100, "current_price": math.nan, "unrealized_pnl": math.nan},
        {"symbol": "9984", "qty": 100, "current_price": 1010.0, "unrealized_pnl": 1000.0},
    ]
    broker.get_short_positions.return_value = [
        {"symbol": "7203", "qty": 100, "current_price": math.nan, "unrealized_pnl": math.nan},
    ]
    rows = dict(_section_paper_balance(broker))
    assert rows["含み損益（買い持ち）"] == "¥1,000"
    assert rows["含み損益（空売り）"] == "¥0"
    assert rows["合計損益"] == "¥2,000"
    assert rows["評価できない建玉数"] == "2"
    assert rows["ロングポジション数"] == "2"
    assert rows["空売りポジション数"] == "1"


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
        run_dashboard(
            recent_days=7,
            drift_n=10,
            analytics=InMemoryAnalyticsQuery(),
            paper_broker=MagicMock(),
        )

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
        run_dashboard(
            recent_days=7,
            drift_n=10,
            analytics=InMemoryAnalyticsQuery(),
            paper_broker=MagicMock(),
        )


if __name__ == "__main__":
    unittest.main()
