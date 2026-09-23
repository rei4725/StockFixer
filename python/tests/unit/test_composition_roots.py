"""合成ルートが本物のアダプタを注入していることの検証。

Phase 4a で生まれた失敗様式への対処。DI 後は「呼び出しがあれば書かれた」が
成り立たず、合成ルートが InMemory 実装を本番側へ貼り間違えても
全テストが緑のまま本番の書き込みだけが止まる。合成ルートそのものを検証する。
"""

import unittest
from unittest.mock import MagicMock, patch

from src.infrastructure.persistence.analytics_query import PostgresAnalyticsQuery
from src.infrastructure.persistence.order_run_repository import PostgresOrderRunSink
from src.infrastructure.persistence.trade_diff_repository import PostgresTradeDiffSink
from src.reporting.types import MonthlyReportSummary


class TestRunAutoTradeCompositionRoot(unittest.TestCase):
    def test_paper_broker_gets_the_injected_sink(self):
        import run_auto_trade

        sink = PostgresTradeDiffSink()
        broker = run_auto_trade.build_broker("paper", sink)

        self.assertIs(broker._trade_diff_sink, sink)

    def test_main_injects_postgres_adapters(self):
        import run_auto_trade

        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return {"buy_orders": 0, "sell_orders": 0, "skipped": 0, "errors": 0}

        with patch("src.trading.execution.run_daily_orders", _capture), patch.object(
            run_auto_trade, "build_broker", return_value=MagicMock()
        ):
            rc = run_auto_trade.main(["--mode", "paper", "--market", "jp"])

        self.assertEqual(rc, 0)
        self.assertIsInstance(captured["order_run_sink"], PostgresOrderRunSink)
        self.assertIsInstance(captured["trade_diff_sink"], PostgresTradeDiffSink)

    def test_broker_and_pipeline_share_one_sink(self):
        """broker に渡した Sink と run_daily_orders に渡す Sink が同一実体であること。"""
        import run_auto_trade

        captured: dict = {}
        seen_sinks: list = []

        def _capture(**kwargs):
            captured.update(kwargs)
            return {"buy_orders": 0, "sell_orders": 0, "skipped": 0, "errors": 0}

        def _build(mode, trade_diff_sink):
            seen_sinks.append(trade_diff_sink)
            return MagicMock()

        with patch("src.trading.execution.run_daily_orders", _capture), patch.object(
            run_auto_trade, "build_broker", _build
        ):
            run_auto_trade.main(["--mode", "paper"])

        self.assertIs(seen_sinks[0], captured["trade_diff_sink"])


class TestDailyJobCompositionRoot(unittest.TestCase):
    def test_run_daily_auto_order_injects_postgres_adapters(self):
        from src.orchestration.jobs import daily

        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return {
                "buy_orders": 0,
                "sell_orders": 0,
                "short_orders": 0,
                "skipped": 0,
                "skipped_min_change": 0,
                "errors": 0,
                "trading_stopped": False,
                "total_turnover": 0.0,
            }

        with patch("src.trading.execution.run_daily_orders", _capture), patch(
            "src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter"
        ):
            try:
                daily.run_daily_auto_order()
            except Exception:
                pass  # Discord 通知など後続処理の失敗はここでは問わない

        self.assertIsInstance(captured.get("order_run_sink"), PostgresOrderRunSink)
        self.assertIsInstance(captured.get("trade_diff_sink"), PostgresTradeDiffSink)

    def test_run_daily_settle_orders_injects_postgres_trade_diff_sink(self):
        """settle_pending_orders() は PaperBroker 経由で唯一 trade diff を書く場所。

        ここに InMemory 実装が紛れ込むと、決済処理は緑のまま本番の
        paper_real_diff 書き込みだけが黙って止まる。
        """
        from src.orchestration.jobs import daily

        captured: dict = {}

        def _fake_paper_broker(**kwargs):
            captured.update(kwargs)
            broker = MagicMock()
            broker.settle_pending_orders.return_value = []
            return broker

        with patch(
            "src.trading.brokers.paper.paper_broker.PaperBroker",
            side_effect=_fake_paper_broker,
        ), patch(
            "src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter"
        ), patch(
            "src.reporting.discord.discord_utils.send_daily_settle_completion"
        ):
            daily.run_daily_settle_orders()

        self.assertIsInstance(captured.get("trade_diff_sink"), PostgresTradeDiffSink)


class TestMonthlyReportJobCompositionRoot(unittest.TestCase):
    def test_run_monthly_report_job_shares_one_postgres_analytics_query(self):
        """run_monthly_report と save_monthly_report_to_file が同一の

        PostgresAnalyticsQuery インスタンスを受け取ることを検証する。
        別々のインスタンスを渡す実装は isinstance だけでは検出できない
        実質的な欠陥（読み取り経路の不整合）になる。
        """
        from src.orchestration.jobs import periodic

        captured: dict = {}

        def _fake_run_monthly_report(*, analytics):
            captured["run_monthly_report_analytics"] = analytics
            return MonthlyReportSummary(
                generated_at="2026-09-24T00:00:00",
                target_month="2026-09",
                net_return=None,
                max_drawdown=None,
                sharpe_ratio=None,
                hit_rate=None,
                avg_slippage=None,
                wf_snapshot_file=None,
                symbol_count=None,
            )

        def _fake_save_monthly_report_to_file(summary, drift_checker=None, *, analytics):
            captured["save_monthly_report_to_file_analytics"] = analytics
            return "results/monthly/2026-09_report.md"

        with patch(
            "src.reporting.monthly.run_monthly_report", side_effect=_fake_run_monthly_report
        ), patch(
            "src.reporting.monthly.save_monthly_report_to_file",
            side_effect=_fake_save_monthly_report_to_file,
        ), patch(
            "src.reporting.discord.discord_utils.send_monthly_report_notification"
        ), patch(
            "src.trading.paper_equity.get_paper_equity_curve",
            return_value=MagicMock(dropna=lambda: []),
        ):
            periodic.run_monthly_report_job()

        analytics_for_run = captured.get("run_monthly_report_analytics")
        analytics_for_save = captured.get("save_monthly_report_to_file_analytics")
        self.assertIsInstance(analytics_for_run, PostgresAnalyticsQuery)
        self.assertIsInstance(analytics_for_save, PostgresAnalyticsQuery)
        self.assertIs(analytics_for_run, analytics_for_save)


if __name__ == "__main__":
    unittest.main()
