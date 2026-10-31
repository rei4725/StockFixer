"""合成ルートが本物のアダプタを注入していることの検証。

Phase 4a で生まれた失敗様式への対処。DI 後は「呼び出しがあれば書かれた」が
成り立たず、合成ルートが InMemory 実装を本番側へ貼り間違えても
全テストが緑のまま本番の書き込みだけが止まる。合成ルートそのものを検証する。
"""

import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.infrastructure.in_memory import InMemoryAnalyticsQuery
from src.infrastructure.persistence.analytics_query import PostgresAnalyticsQuery
from src.infrastructure.persistence.order_run_repository import PostgresOrderRunSink
from src.infrastructure.persistence.prediction_result_repository import (
    PostgresPredictionResultRepository,
)
from src.infrastructure.persistence.trade_diff_repository import PostgresTradeDiffSink
from src.reporting.types import MonthlyReportSummary
from src.rule_engine.types import RuleSignalBatch


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
        self.assertIsInstance(captured["prediction_repo"], PostgresPredictionResultRepository)

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

    def test_main_passes_prediction_saved_marker(self):
        """CLI も保存の終わりの印を run_daily_orders に渡す（#1443）"""
        import run_auto_trade

        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return {"buy_orders": 0, "sell_orders": 0, "skipped": 0, "errors": 0}

        with patch("src.trading.execution.run_daily_orders", _capture), patch.object(
            run_auto_trade, "build_broker", return_value=MagicMock()
        ), patch(
            "src.orchestration.jobs.prediction_saved_marker.read_prediction_saved_day",
            return_value="20261030",
        ):
            rc = run_auto_trade.main(["--mode", "paper"])

        self.assertEqual(rc, 0)
        self.assertEqual(captured["prediction_saved_day"], "20261030")

    def test_main_without_marker_blocks_new_and_keeps_sl(self):
        """印が無い朝に CLI で流しても、新規の買いは出ず SL の決済は出る（#1443）"""
        import pandas as pd
        import run_auto_trade

        from src.infrastructure.in_memory import (
            InMemoryOrderRunSink,
            InMemoryPredictionRepository,
            InMemoryTradeDiffSink,
        )
        from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
        from src.trading.types import CorrelationGateResult, TradingGateStatus

        today_run = "20261029_224343"  # 7:43 JST の今日の回
        predictions = pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "7200",
                    "predicted_at": today_run,
                    "current_price": 1000.0,
                    "diff_ratio": 0.03,
                    "confidence_ratio": 1.0,
                }
            ]
        )
        repo = InMemoryPredictionRepository()
        broker = MagicMock(spec=BrokerBase)
        broker.broker_name = "paper"
        broker.get_balance.return_value = 1_000_000.0
        broker.get_positions.return_value = [
            {"symbol": "9000", "qty": 100, "avg_price": 1000.0, "current_price": 900.0}
        ]
        broker.get_short_positions = MagicMock(return_value=[])
        broker.send_order.return_value = {"order_id": "test-001", "status": "pending"}
        sl_risk = MagicMock()
        sl_risk.return_value.check_sl_tp.side_effect = lambda avg, cur: (
            (cur - avg) / avg <= -0.05,
            False,
            "SL発動",
        )
        open_gate = TradingGateStatus(
            is_allowed=True, stop_active=False, daily_loss=0.0, daily_loss_limit=20_000.0
        )
        with ExitStack() as stack:

            def p(*args, **kwargs):
                return stack.enter_context(patch(*args, **kwargs))

            stack.enter_context(patch.object(run_auto_trade, "build_broker", return_value=broker))
            stack.enter_context(
                patch.object(run_auto_trade, "PostgresTradeDiffSink", InMemoryTradeDiffSink)
            )
            stack.enter_context(
                patch.object(run_auto_trade, "PostgresOrderRunSink", InMemoryOrderRunSink)
            )
            stack.enter_context(
                patch.object(
                    run_auto_trade, "PostgresPredictionResultRepository", return_value=repo
                )
            )
            stack.enter_context(
                patch.object(repo, "get_latest_by_market", return_value=predictions)
            )
            p(
                "src.orchestration.jobs.prediction_saved_marker.read_prediction_saved_day",
                return_value="",
            )
            p(
                "src.trading.execution.freshness._local_now",
                return_value=datetime(2026, 10, 29, 23, 50, tzinfo=timezone.utc),
            )
            p("src.trading.execution.runner._record_order")
            p("src.trading.execution.sl_tp._record_order")
            p(
                "src.trading.execution.runner._choose_order_params",
                return_value=(OrderType.MARKET, 0.0, "market", "open"),
            )
            p("src.trading.execution.sl_tp.RiskManager", sl_risk)
            p("src.trading.execution.RiskManager.evaluate_trading_gate", return_value=open_gate)
            p("src.trading.execution.RiskManager.calc_position_size", return_value=100)
            p("src.trading.execution.RiskManager.update_peak_balance")
            p(
                "src.trading.execution.runner.evaluate_correlation_gate",
                return_value=CorrelationGateResult(
                    is_allowed=True, enc=2.0, enc_threshold=2.0, avg_correlation=0.0, n_symbols=1
                ),
            )
            p("src.trading.execution.runner._load_exit_model", return_value=None)
            p("src.trading.execution.params.get_optimal_params", return_value={})
            p(
                "src.trading.execution.selection.get_symbol_sector",
                side_effect=lambda market, symbol: f"UNKNOWN:{market}:{symbol}",
            )
            rc = run_auto_trade.main(["--mode", "paper"])

        self.assertEqual(rc, 0)
        sent = [(c.args[0], c.args[1]) for c in broker.send_order.call_args_list]
        self.assertEqual(sent, [("9000", OrderSide.SELL)])


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
        self.assertIsInstance(captured.get("prediction_repo"), PostgresPredictionResultRepository)

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
            broker.last_settle_errors = 0
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


class TestRunHorizonExitCheckCompositionRoot(unittest.TestCase):
    def test_run_horizon_exit_check_injects_postgres_trade_diff_sink(self):
        from src.orchestration.jobs import daily

        captured: dict = {}

        def _fake_paper_broker(**kwargs):
            captured.update(kwargs)
            broker = MagicMock()
            broker.get_positions.return_value = []
            return broker

        fake_con = MagicMock()
        fake_con.execute.return_value.fetchall.return_value = [("AAPL",)]
        fake_cm = MagicMock()
        fake_cm.__enter__.return_value = fake_con
        fake_cm.__exit__.return_value = False

        with patch(
            "src.trading.brokers.paper.paper_broker.PaperBroker",
            side_effect=_fake_paper_broker,
        ), patch(
            "src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter"
        ), patch(
            "src.utils.db._connection._db_connection", return_value=fake_cm
        ), patch.dict(
            "os.environ", {"AUTO_TRADE_MODE": "paper"}
        ):
            daily.run_horizon_exit_check()

        self.assertIsInstance(captured.get("trade_diff_sink"), PostgresTradeDiffSink)


class TestRunDailyPaperTradeReportCompositionRoot(unittest.TestCase):
    def test_run_daily_paper_trade_report_injects_postgres_trade_diff_sink(self):
        from src.orchestration.jobs import daily

        captured: dict = {}

        def _fake_paper_broker(**kwargs):
            captured.update(kwargs)
            broker = MagicMock()
            broker.get_positions.return_value = []
            broker.get_pnl_summary.return_value = {}
            return broker

        with patch(
            "src.trading.brokers.paper.paper_broker.PaperBroker",
            side_effect=_fake_paper_broker,
        ), patch(
            "src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter"
        ), patch(
            "src.reporting.discord.discord_utils.send_paper_trade_position_report"
        ), patch.dict(
            "os.environ", {"AUTO_TRADE_MODE": "paper"}
        ):
            daily.run_daily_paper_trade_report()

        self.assertIsInstance(captured.get("trade_diff_sink"), PostgresTradeDiffSink)


class TestRunPreCloseAlertCompositionRoot(unittest.TestCase):
    def test_run_pre_close_alert_injects_postgres_trade_diff_sink(self):
        from src.orchestration.jobs import daily

        captured: dict = {}

        def _fake_get_pre_close_alerts(**kwargs):
            captured.update(kwargs)
            return []

        with patch(
            "src.trading.pre_close_alert_service.get_pre_close_alerts",
            side_effect=_fake_get_pre_close_alerts,
        ), patch(
            "src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter"
        ), patch(
            "src.reporting.discord.discord_utils.send_webhook_notification"
        ), patch.dict(
            "os.environ", {"AUTO_TRADE_MODE": "paper"}
        ):
            daily.run_pre_close_alert()

        self.assertIsInstance(captured.get("trade_diff_sink"), PostgresTradeDiffSink)
        # 旧実装はプロキシ経由で予測を読んでいた。結線漏れは実行時まで露見しないため固定する
        self.assertIsInstance(captured.get("prediction_repo"), PostgresPredictionResultRepository)


class TestRunDailyRuleSignalsCompositionRoot(unittest.TestCase):
    def test_run_daily_rule_signals_injects_postgres_trade_diff_sink(self):
        from src.orchestration.jobs import daily

        captured: dict = {}

        def _fake_execute_rule_paper_trades(**kwargs):
            captured.update(kwargs)
            return {"buy_orders": 0, "sell_orders": 0}

        with patch(
            "src.rule_engine.pipeline.generate_rule_signals", return_value=RuleSignalBatch()
        ), patch(
            "src.trading.rule_execution.execute_rule_paper_trades",
            side_effect=_fake_execute_rule_paper_trades,
        ), patch(
            "src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter"
        ), patch(
            "src.reporting.discord.discord_utils.send_rule_daily_signals"
        ):
            daily.run_daily_rule_signals()

        self.assertIsInstance(captured.get("trade_diff_sink"), PostgresTradeDiffSink)


class TestRunClaudeTraderCompositionRoot(unittest.TestCase):
    def test_main_injects_postgres_trade_diff_sink(self):
        import run_claude_trader

        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return {"buy_orders": 0, "sell_orders": 0, "skipped": 0, "errors": 0}

        with patch("config.settings.CLAUDE_TRADER_ENABLED", True), patch(
            "src.trading.claude_agent.run_claude_trader", side_effect=_capture
        ), patch.object(run_claude_trader, "build_broker", return_value=MagicMock()):
            rc = run_claude_trader.main(["--mode", "paper", "--market", "jp"])

        self.assertEqual(rc, 0)
        self.assertIsInstance(captured.get("trade_diff_sink"), PostgresTradeDiffSink)
        self.assertIsInstance(captured.get("prediction_repo"), PostgresPredictionResultRepository)


class TestRunRuleSignalsCompositionRoot(unittest.TestCase):
    def test_main_injects_postgres_trade_diff_sink(self):
        import run_rule_signals

        captured: dict = {}

        def _fake_execute_rule_paper_trades(**kwargs):
            captured.update(kwargs)
            return {"buy_orders": 0, "sell_orders": 0}

        fake_signal = {
            "symbol": "AAPL",
            "rule": "dummy_rule",
            "price": 100.0,
            "win_rate": 0.5,
            "net_profit": 0.0,
            "signal": 1,
        }

        with patch("run_rule_signals.YFinanceMarketDataAdapter"), patch(
            "run_rule_signals.generate_rule_signals",
            return_value=RuleSignalBatch(signals=[fake_signal], total=1),
        ), patch("run_rule_signals.make_order_broker"), patch(
            "run_rule_signals.execute_rule_paper_trades",
            side_effect=_fake_execute_rule_paper_trades,
        ), patch(
            "sys.argv", ["run_rule_signals.py", "--market", "jp"]
        ):
            run_rule_signals.main()

        self.assertIsInstance(captured.get("trade_diff_sink"), PostgresTradeDiffSink)


class TestRunDashboardCompositionRoot(unittest.TestCase):
    def test_main_injects_postgres_analytics_query(self):
        import run_dashboard

        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)

        with patch.object(run_dashboard, "run_dashboard", side_effect=_capture), patch(
            "src.orchestration.port_wiring.wire_ports"
        ), patch("sys.argv", ["run_dashboard.py"]):
            run_dashboard.main()

        self.assertIsInstance(captured.get("analytics"), PostgresAnalyticsQuery)
        # ペーパーの含み損益は PaperBroker の評価から出す（#1178）
        from src.trading.brokers.paper.paper_broker import PaperBroker

        self.assertIsInstance(captured.get("paper_broker"), PaperBroker)


class TestRunMonthlyReportCliCompositionRoot(unittest.TestCase):
    def test_main_injects_postgres_analytics_query(self):
        import run_monthly_report

        captured: dict = {}

        def _capture(*, target_month, analytics):
            captured["analytics"] = analytics
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

        with patch.object(run_monthly_report, "run_monthly_report", side_effect=_capture), patch(
            "src.orchestration.port_wiring.wire_ports"
        ), patch("sys.argv", ["run_monthly_report.py"]):
            run_monthly_report.main()

        self.assertIsInstance(captured.get("analytics"), PostgresAnalyticsQuery)


class TestRunWeeklyReportCompositionRoot(unittest.TestCase):
    def test_run_weekly_report_uses_postgres_analytics_query_for_diff_summary(self):
        """diff_summary が PostgresAnalyticsQuery から得られたものであることを検証する。

        run_weekly_report は analytics= を渡す形ではなく、その場で
        PostgresAnalyticsQuery().paper_real_diff_summary() を呼んで結果を
        send_weekly_report の diff_summary 引数に渡す。InMemory 実装が
        紛れ込むと send_weekly_report は呼ばれ続けるが値が常にゼロになる。
        """
        from src.orchestration.jobs import weekly

        sentinel_diff_summary = {"tracked_count": 42}
        captured: dict = {}

        fake_query_instance = MagicMock()
        fake_query_instance.paper_real_diff_summary.return_value = sentinel_diff_summary

        def _fake_send_weekly_report(**kwargs):
            captured.update(kwargs)

        with patch(
            "src.infrastructure.persistence.analytics_query.PostgresAnalyticsQuery",
            return_value=fake_query_instance,
        ) as mock_cls, patch("src.prediction.db.save_weekly_accuracy_snapshot"), patch(
            "src.reporting.discord.discord_utils.send_weekly_report",
            side_effect=_fake_send_weekly_report,
        ), patch(
            "src.reporting.llm_review.generate_weekly_review", return_value=None
        ), patch(
            "src.prediction.db.load_top_prediction_misses", return_value=MagicMock()
        ), patch(
            "src.prediction.miss_analysis.run_miss_analysis_batch", return_value=[]
        ), patch(
            "src.reporting.discord.discord_utils.send_miss_analysis_summary"
        ):
            weekly.run_weekly_report()

        mock_cls.assert_called_once_with()
        fake_query_instance.paper_real_diff_summary.assert_called_once_with(recent_days=7)
        self.assertIs(captured.get("diff_summary"), sentinel_diff_summary)

    def test_run_weekly_report_reads_snapshots_after_saving_this_week(self):
        """前週比較用スナップショットは今週分を保存した後に読み、そのまま渡すこと。

        以前は send_weekly_report が自分で読んでいたため、保存→読み取りの順序は
        呼び出し順から暗黙に保証されていた。入口へ押し上げた今は入口が保証する。
        """
        from src.orchestration.jobs import weekly

        order: list[str] = []
        drift_df = MagicMock(name="drift_df")
        snapshots_df = MagicMock(name="snapshots_df")
        fake_query = MagicMock()
        fake_query.paper_real_diff_summary.return_value = {"tracked_count": 0}

        def _snapshots(**kwargs):
            order.append("load_snapshots")
            return snapshots_df

        fake_query.weekly_accuracy_snapshots.side_effect = _snapshots
        week_df = MagicMock(name="week_df")
        captured: dict = {}

        with patch(
            "src.infrastructure.persistence.analytics_query.PostgresAnalyticsQuery",
            return_value=fake_query,
        ), patch(
            "src.prediction.db.load_drift_summary", return_value=drift_df
        ) as mock_drift, patch(
            "src.prediction.db.load_weekly_accuracy_summary", return_value=week_df
        ) as mock_week, patch(
            "src.prediction.db.save_weekly_accuracy_snapshot",
            side_effect=lambda *a, **k: order.append("save_snapshot"),
        ) as mock_save, patch(
            "src.reporting.discord.discord_utils.send_weekly_report",
            side_effect=lambda **kwargs: captured.update(kwargs),
        ), patch(
            "src.reporting.llm_review.generate_weekly_review", return_value=None
        ), patch(
            "src.prediction.db.load_top_prediction_misses", return_value=MagicMock()
        ), patch(
            "src.prediction.miss_analysis.run_miss_analysis_batch", return_value=[]
        ), patch(
            "src.reporting.discord.discord_utils.send_miss_analysis_summary"
        ):
            weekly.run_weekly_report()

        self.assertEqual(order, ["save_snapshot", "load_snapshots"])
        # 読み込みの失敗を失敗の知らせに届けるため、週次レポートの道だけ投げる（#1305）
        mock_drift.assert_called_once_with(horizon=1, model_name="production", raise_on_error=True)
        fake_query.weekly_accuracy_snapshots.assert_called_once_with(n_weeks=4)
        # 週の記録は今週に予想した採点の行だけで作る（#1023）
        week_start = mock_save.call_args.args[0]
        mock_week.assert_called_once_with(
            week_start, horizon=1, model_name="production", raise_on_error=True
        )
        self.assertIs(mock_save.call_args.args[1], week_df)
        self.assertIs(captured.get("accuracy_df"), drift_df)
        self.assertIs(captured.get("snapshots_df"), snapshots_df)


class TestExternalV1PredictionsCompositionRoot(unittest.TestCase):
    def test_top_predictions_endpoint_injects_postgres_repository(self):
        from src.api.health import app

        captured: dict = {}

        def _capture(*, predictions):
            captured["predictions"] = predictions
            return None, []

        app.config["TESTING"] = True
        with patch(
            "src.api.external_v1._load_api_keys", return_value=frozenset(["test-key"])
        ), patch("src.api.external_v1._is_rate_limited", return_value=False), patch(
            "src.reporting.query_service.get_latest_market_prediction_snapshots",
            side_effect=_capture,
        ):
            with app.test_client() as client:
                resp = client.get("/api/v1/predictions/top", headers={"X-API-Key": "test-key"})

        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(captured.get("predictions"), PostgresPredictionResultRepository)


class TestExternalV1MonthlyReportCompositionRoot(unittest.TestCase):
    def test_monthly_report_endpoint_injects_postgres_analytics_query(self):
        from src.api.health import app

        captured: dict = {}

        def _capture(*, target_month, analytics):
            captured["analytics"] = analytics
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

        app.config["TESTING"] = True
        with patch(
            "src.api.external_v1._load_api_keys", return_value=frozenset(["test-key"])
        ), patch("src.api.external_v1._is_rate_limited", return_value=False), patch(
            "src.reporting.query_service.get_monthly_report_summary", side_effect=_capture
        ):
            with app.test_client() as client:
                resp = client.get("/api/v1/reports/monthly", headers={"X-API-Key": "test-key"})

        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(captured.get("analytics"), PostgresAnalyticsQuery)


class TestDiscordBotMonthlyReportCompositionRoot(unittest.TestCase):
    def test_wire_ports_registers_postgres_analytics_query(self):
        """Bot の 2 つの入口（run_discord_bot / run_scheduler --with-bot）が呼ぶ
        wire_ports() が、Bot の使う AnalyticsQuery として Postgres 実装を注入すること。
        """
        from src.orchestration import port_wiring
        from src.reporting import ports as reporting_ports

        saved_wired = port_wiring._wired
        try:
            reporting_ports._analytics_query = None
            port_wiring.wire_ports(force=True)
            self.assertIsInstance(reporting_ports.get_analytics_query(), PostgresAnalyticsQuery)
        finally:
            port_wiring._wired = saved_wired

    def test_wire_ports_registers_postgres_prediction_result_repository(self):
        from src.orchestration import port_wiring
        from src.reporting import ports as reporting_ports

        saved_wired = port_wiring._wired
        try:
            reporting_ports._prediction_results = None
            port_wiring.wire_ports(force=True)
            self.assertIsInstance(
                reporting_ports.get_prediction_result_repository(),
                PostgresPredictionResultRepository,
            )
        finally:
            port_wiring._wired = saved_wired

    def test_forecast_and_ranking_use_injected_repository(self):
        import asyncio

        from src.infrastructure.in_memory import InMemoryPredictionRepository
        from src.reporting.discord import discord_bot
        from src.reporting.ports import set_prediction_result_repository

        repo = InMemoryPredictionRepository()
        repo.add(
            "20260927_090000",
            [
                {
                    "market": "jp",
                    "symbol": "7203",
                    "current_price": 100.0,
                    "avg_pred_price": 101.0,
                    "diff_ratio": 0.01,
                    "model_count": 2,
                }
            ],
        )
        set_prediction_result_repository(repo)

        self.assertIn("7203", discord_bot.get_top10_diff_stocks_message("jp", "top10"))

        sent: list = []
        message = MagicMock()

        async def _send(content=None, **kwargs):
            sent.append(content)

        message.channel.send = _send
        asyncio.run(discord_bot.handle_forecast_command(message))
        self.assertTrue(any("7203" in str(c) for c in sent))

    def test_bot_entrypoints_call_wire_ports(self):
        """Bot を起動する合成ルートが wire_ports() を呼んでいること（未注入なら
        /monthlyreport が RuntimeError でしか失敗を知らせないため、ここで固定する）。
        """
        root = Path(__file__).resolve().parents[2]
        for name in ("run_discord_bot.py", "run_scheduler.py"):
            with self.subTest(entrypoint=name):
                self.assertIn("wire_ports()", (root / name).read_text(encoding="utf-8"))

    def test_handle_monthlyreport_command_uses_injected_analytics_query(self):
        import asyncio

        from src.reporting.discord.discord_bot import handle_monthlyreport_command
        from src.reporting.ports import set_analytics_query

        sentinel = InMemoryAnalyticsQuery()
        set_analytics_query(sentinel)

        captured: dict = {}

        def _capture(target_month, *, analytics):
            captured["analytics"] = analytics
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

        message = MagicMock()
        message.content = "/monthlyreport"
        message.channel = MagicMock()

        async def _send(*args, **kwargs):
            return None

        message.channel.send = _send

        with patch(
            "src.reporting.discord.discord_bot.get_monthly_report_summary",
            side_effect=_capture,
        ):
            asyncio.run(handle_monthlyreport_command(message))

        self.assertIs(captured.get("analytics"), sentinel)


if __name__ == "__main__":
    unittest.main()


class TestTextReviewPortCompositionRoots(unittest.TestCase):
    """LLM を使う 4 つの合成ルートが factory 製の TextReviewPort を注入すること（#741）。

    BC は factory を直接呼ばなくなったため、合成ルートが渡し忘れると
    TypeError で落ちる。渡す物が factory 製であることをここで固定する。
    """

    def test_run_weekly_report_injects_factory_port(self):
        from src.orchestration.jobs import weekly

        sentinel_port = MagicMock(name="text_review_port")
        with patch(
            "src.infrastructure.llm.factory.get_text_review_port", return_value=sentinel_port
        ), patch("src.infrastructure.persistence.analytics_query.PostgresAnalyticsQuery"), patch(
            "src.prediction.db.save_weekly_accuracy_snapshot"
        ), patch(
            "src.reporting.discord.discord_utils.send_weekly_report"
        ), patch(
            "src.reporting.llm_review.generate_weekly_review", return_value=None
        ) as mock_review, patch(
            "src.prediction.db.load_top_prediction_misses", return_value=MagicMock()
        ), patch(
            "src.prediction.miss_analysis.run_miss_analysis_batch", return_value=[]
        ), patch(
            "src.reporting.discord.discord_utils.send_miss_analysis_summary"
        ):
            weekly.run_weekly_report()

        mock_review.assert_called_once()
        self.assertIs(mock_review.call_args.kwargs["review_port"], sentinel_port)

    def test_run_nightly_strategy_factory_injects_factory_port(self):
        from src.domain.types import SymbolTask
        from src.orchestration.jobs import periodic

        sentinel_port = MagicMock(name="text_review_port")
        with patch(
            "src.infrastructure.llm.factory.get_text_review_port", return_value=sentinel_port
        ), patch(
            "src.watchlist.batch_runner.load_target_symbols",
            return_value=[SymbolTask(market="jp", symbol="7203")],
        ), patch(
            "src.backtest.factory.run_factory_batch"
        ) as mock_batch, patch(
            "src.reporting.discord.discord_utils.send_factory_completion"
        ):
            periodic.run_nightly_strategy_factory(market="jp", budget=1, force=True)

        mock_batch.assert_called_once()
        self.assertIs(mock_batch.call_args.kwargs["review_port"], sentinel_port)

    def test_run_backtest_review_main_injects_factory_port(self):
        import run_backtest_review

        sentinel_port = MagicMock(name="text_review_port")
        with patch.object(
            run_backtest_review, "get_text_review_port", return_value=sentinel_port
        ), patch.object(
            run_backtest_review, "run_backtest_review", return_value=[]
        ) as mock_run, patch(
            "sys.argv", ["run_backtest_review.py", "--dry-run"]
        ):
            run_backtest_review.main()

        self.assertIs(mock_run.call_args.kwargs["review_port"], sentinel_port)

    def test_run_test_gap_main_injects_factory_port(self):
        import run_test_gap

        sentinel_port = MagicMock(name="text_review_port")
        with patch.object(
            run_test_gap, "get_text_review_port", return_value=sentinel_port
        ), patch.object(run_test_gap, "run_test_gap_review", return_value=[]) as mock_run, patch(
            "sys.argv", ["run_test_gap.py", "--dry-run"]
        ):
            run_test_gap.main()

        self.assertIs(mock_run.call_args.kwargs["review_port"], sentinel_port)
