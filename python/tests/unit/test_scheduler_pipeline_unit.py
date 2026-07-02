"""ユニットテスト: scheduler_pipeline の自動発注連携"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from src.domain.types import SymbolTask
from src.orchestration.scheduler import run_daily_auto_order, run_daily_drift_check


class TestRunDailyAutoOrder(unittest.TestCase):
    @patch("src.reporting.discord.discord_utils.send_daily_order_completion")
    @patch("src.trading.execution.run_daily_orders")
    @patch("src.trading.brokers.paper.paper_broker.PaperBroker")
    def test_paper_mode_forwards_stop_fields_to_notification(
        self,
        mock_paper_broker,
        mock_run_daily_orders,
        mock_send_completion,
    ):
        broker = MagicMock()
        mock_paper_broker.return_value = broker
        mock_run_daily_orders.return_value = {
            "buy_orders": 0,
            "sell_orders": 0,
            "skipped": 0,
            "errors": 0,
            "trading_stopped": True,
            "stop_reason": "日次損失上限に到達",
            "daily_loss": 25_000.0,
            "daily_loss_limit": 20_000.0,
        }

        with patch.dict("os.environ", {"AUTO_TRADE_MODE": "paper"}, clear=False):
            run_daily_auto_order()

        call_kwargs = mock_run_daily_orders.call_args.kwargs
        self.assertEqual(call_kwargs["broker"], broker)
        self.assertEqual(call_kwargs["market"], "jp")
        self.assertEqual(call_kwargs["mode"], "paper")
        self.assertIn("market_data", call_kwargs)
        mock_send_completion.assert_called_once_with(
            buy_orders=0,
            sell_orders=0,
            mode="paper",
            trading_stopped=True,
            stop_reason="日次損失上限に到達",
            daily_loss=25_000.0,
            daily_loss_limit=20_000.0,
        )

    @patch("src.reporting.discord.discord_utils.send_daily_order_completion")
    @patch("src.trading.execution.run_daily_orders")
    def test_live_mode_uses_kabu_broker(
        self,
        mock_run_daily_orders,
        mock_send_completion,
    ):
        broker = MagicMock()
        mock_kabu_broker = MagicMock(return_value=broker)
        mock_run_daily_orders.return_value = {
            "buy_orders": 1,
            "sell_orders": 0,
            "skipped": 0,
            "errors": 0,
            "trading_stopped": False,
            "stop_reason": None,
            "daily_loss": 0.0,
            "daily_loss_limit": 20_000.0,
        }

        fake_kabu_module = SimpleNamespace(KabuBroker=mock_kabu_broker)
        with (
            patch.dict("os.environ", {"AUTO_TRADE_MODE": "live"}, clear=False),
            patch.dict(sys.modules, {"src.trading.brokers.kabu.kabu_client": fake_kabu_module}),
        ):
            run_daily_auto_order()

        call_kwargs = mock_run_daily_orders.call_args.kwargs
        self.assertEqual(call_kwargs["broker"], broker)
        self.assertEqual(call_kwargs["market"], "jp")
        self.assertEqual(call_kwargs["mode"], "live")
        mock_send_completion.assert_called_once()


class TestRunDailyDriftCheck(unittest.TestCase):
    @patch("src.prediction.training_pipeline.train_models_for_symbol_task")
    @patch("src.watchlist.batch_runner.load_target_symbols")
    @patch("src.reporting.discord.discord_utils.send_drift_retrain_notification")
    @patch("src.prediction.db.load_drift_summary")
    def test_retrain_runs_for_triggered_symbols(
        self,
        mock_load_summary,
        mock_notify,
        mock_load_symbols,
        mock_train,
    ):
        mock_load_summary.return_value = pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "7203",
                    "mean_abs_error": 0.03,
                    "direction_accuracy": 0.40,
                },
                {
                    "market": "jp",
                    "symbol": "7201",
                    "mean_abs_error": 0.01,
                    "direction_accuracy": 0.60,
                },
            ]
        )
        mock_load_symbols.return_value = [SymbolTask(market="jp", symbol="7203")]
        mock_train.return_value = {"status": "success"}

        run_daily_drift_check()

        mock_notify.assert_called_once()
        mock_train.assert_called_once()
        task = mock_train.call_args.args[0]
        self.assertEqual(task.market, "jp")
        self.assertEqual(task.symbol, "7203")

    @patch("src.prediction.training_pipeline.train_models_for_symbol_task")
    @patch("src.watchlist.batch_runner.load_target_symbols")
    @patch("src.reporting.discord.discord_utils.send_drift_retrain_notification")
    @patch("src.prediction.db.load_drift_summary")
    def test_no_retrain_when_summary_is_empty(
        self,
        mock_load_summary,
        mock_notify,
        mock_load_symbols,
        mock_train,
    ):
        mock_load_summary.return_value = pd.DataFrame()

        run_daily_drift_check()

        mock_notify.assert_not_called()
        mock_load_symbols.assert_not_called()
        mock_train.assert_not_called()

    @patch("src.reporting.discord.discord_utils.send_feature_suggestion_notification")
    @patch("src.prediction.db.load_feature_exclusion_candidates")
    @patch("src.prediction.training_pipeline.train_models_for_symbol_task")
    @patch("src.watchlist.batch_runner.load_target_symbols")
    @patch("src.reporting.discord.discord_utils.send_drift_retrain_notification")
    @patch("src.prediction.db.load_drift_summary")
    def test_feature_suggestion_sent_after_successful_retrain(
        self,
        mock_load_summary,
        mock_notify_drift,
        mock_load_symbols,
        mock_train,
        mock_load_candidates,
        mock_notify_suggestions,
    ):
        mock_load_summary.return_value = pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "7203",
                    "mean_abs_error": 0.03,
                    "direction_accuracy": 0.40,
                }
            ]
        )
        mock_load_symbols.return_value = [SymbolTask(market="jp", symbol="7203")]
        mock_train.return_value = {"status": "success", "shap_results": []}
        mock_load_candidates.return_value = pd.DataFrame(
            [{"feature": "rsi", "importance_mean": 0.001, "importance_rank": 50}]
        )

        run_daily_drift_check()

        mock_load_candidates.assert_called_once_with("jp", "7203")
        mock_notify_suggestions.assert_called_once()

    @patch("src.reporting.discord.discord_utils.send_feature_suggestion_notification")
    @patch("src.prediction.db.load_feature_exclusion_candidates")
    @patch("src.prediction.training_pipeline.train_models_for_symbol_task")
    @patch("src.watchlist.batch_runner.load_target_symbols")
    @patch("src.reporting.discord.discord_utils.send_drift_retrain_notification")
    @patch("src.prediction.db.load_drift_summary")
    def test_feature_suggestion_not_sent_when_retrain_failed(
        self,
        mock_load_summary,
        mock_notify_drift,
        mock_load_symbols,
        mock_train,
        mock_load_candidates,
        mock_notify_suggestions,
    ):
        mock_load_summary.return_value = pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "7203",
                    "mean_abs_error": 0.03,
                    "direction_accuracy": 0.40,
                }
            ]
        )
        mock_load_symbols.return_value = [SymbolTask(market="jp", symbol="7203")]
        mock_train.return_value = {"status": "error", "error": "学習失敗"}

        run_daily_drift_check()

        mock_load_candidates.assert_not_called()
        mock_notify_suggestions.assert_not_called()

    @patch("src.prediction.training_pipeline.train_models_for_symbol_task")
    @patch("src.watchlist.batch_runner.load_target_symbols")
    @patch("src.reporting.discord.discord_utils.send_drift_retrain_notification")
    @patch("src.prediction.db.load_drift_summary")
    def test_no_retrain_when_no_symbol_crosses_threshold(
        self,
        mock_load_summary,
        mock_notify,
        mock_load_symbols,
        mock_train,
    ):
        mock_load_summary.return_value = pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "7203",
                    "mean_abs_error": 0.01,
                    "direction_accuracy": 0.60,
                }
            ]
        )

        run_daily_drift_check()

        mock_notify.assert_not_called()
        mock_load_symbols.assert_not_called()
        mock_train.assert_not_called()


if __name__ == "__main__":
    unittest.main()


# ──────────────────────────────────────────────────────────────
# 追加: 週次・日次ジョブ関数テスト
# ──────────────────────────────────────────────────────────────


def _mock_anthropic(text="(1)総評 (2)懸念 (3)推奨"):
    """anthropic.Anthropic を差し替えるモックを返す。"""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    response = MagicMock()
    response.content = [text_block]
    client = MagicMock()
    client.messages.create.return_value = response
    anthropic_module = MagicMock()
    anthropic_module.Anthropic.return_value = client
    return anthropic_module, client


# 環境変数 LLM_BACKEND に依存させず、anthropic モックが効く SDK 経路に固定する
# （cli だと factory が実 claude CLI を起動してしまう）
@patch("src.infrastructure.llm.factory.LLM_BACKEND", "sdk")
class TestRunWeeklyReport(unittest.TestCase):
    """run_weekly_report のテスト"""

    @patch("src.reporting.discord.discord_utils.send_weekly_report")
    @patch("src.prediction.db.load_paper_real_diff_summary")
    @patch("src.prediction.db.load_drift_summary")
    def test_calls_send_weekly_report(self, mock_drift, mock_diff, mock_send):
        from src.orchestration.scheduler import run_weekly_report

        mock_drift.return_value = pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "7203",
                    "direction_accuracy": 0.6,
                    "mean_abs_error": 0.01,
                    "n_samples": 30,
                }
            ]
        )
        mock_diff.return_value = {
            "tracked_count": 12,
            "comparable_count": 8,
            "avg_paper_slippage": 0.001,
            "avg_real_slippage": 0.002,
            "avg_abs_price_diff": 1.5,
            "avg_abs_diff_ratio": 0.0015,
        }
        anthropic_module, _ = _mock_anthropic()
        with patch.dict(sys.modules, {"anthropic": anthropic_module}):
            run_weekly_report()
        mock_send.assert_called_once()

    @patch("src.reporting.discord.discord_utils.send_weekly_report")
    @patch("src.prediction.db.load_paper_real_diff_summary")
    @patch("src.prediction.db.load_drift_summary")
    def test_exception_does_not_propagate(self, mock_drift, mock_diff, mock_send):
        from src.orchestration.scheduler import run_weekly_report

        mock_drift.side_effect = Exception("DB エラー")
        run_weekly_report()  # 例外が外に出ないこと


class TestRunDailySettle(unittest.TestCase):
    """run_daily_settle_orders のテスト"""

    @patch("src.reporting.discord.discord_utils.send_daily_settle_completion")
    @patch("src.trading.brokers.paper.paper_broker.PaperBroker")
    def test_paper_mode_calls_settle(self, mock_broker_cls, mock_send):
        from unittest.mock import patch as _patch

        from src.orchestration.scheduler import run_daily_settle_orders

        mock_broker = mock_broker_cls.return_value
        mock_broker.settle_pending_orders.return_value = [{"symbol": "7203", "qty": 10}]
        with _patch.dict("os.environ", {"AUTO_TRADE_MODE": "paper"}):
            run_daily_settle_orders()
        mock_broker.settle_pending_orders.assert_called()

    @patch("src.reporting.discord.discord_utils.send_daily_settle_completion")
    def test_live_mode_is_skipped(self, mock_send):
        from unittest.mock import patch as _patch

        from src.orchestration.scheduler import run_daily_settle_orders

        with _patch.dict("os.environ", {"AUTO_TRADE_MODE": "live"}):
            run_daily_settle_orders()
        mock_send.assert_not_called()


class TestRunWeeklyWalkForwardReport(unittest.TestCase):
    """run_weekly_walk_forward_report のテスト"""

    @patch("src.reporting.discord.discord_utils.send_walk_forward_report_completion")
    @patch("src.backtest.walk_forward_report.run_walk_forward_comparison_report")
    @patch("src.watchlist.batch_runner.load_target_symbols")
    def test_sends_completion_notification(self, mock_symbols, mock_run, mock_send):
        from src.orchestration.scheduler import run_weekly_walk_forward_report

        mock_symbols.return_value = []
        mock_run.return_value = {"success": 5, "failed": 1, "total": 6}
        run_weekly_walk_forward_report()
        mock_send.assert_called_once()

    @patch("src.reporting.discord.discord_utils.send_walk_forward_report_completion")
    @patch("src.backtest.walk_forward_report.run_walk_forward_comparison_report")
    @patch("src.watchlist.batch_runner.load_target_symbols")
    def test_exception_does_not_propagate(self, mock_symbols, mock_run, mock_send):
        from src.orchestration.scheduler import run_weekly_walk_forward_report

        mock_symbols.return_value = []
        mock_run.side_effect = Exception("レポートエラー")
        run_weekly_walk_forward_report()  # 例外が外に出ないこと


class TestRunWeeklyWatchlistRefresh(unittest.TestCase):
    """run_weekly_watchlist_refresh のテスト"""

    @patch("src.reporting.discord.discord_utils.send_watchlist_update_report")
    @patch("src.watchlist.manager.run_watchlist_refresh")
    def test_sends_update_report(self, mock_refresh, mock_send):
        from src.orchestration.scheduler import run_weekly_watchlist_refresh

        mock_refresh.return_value = []
        run_weekly_watchlist_refresh()
        mock_send.assert_called_once()

    @patch("src.reporting.discord.discord_utils.send_watchlist_update_report")
    @patch("src.watchlist.manager.run_watchlist_refresh")
    def test_exception_does_not_propagate(self, mock_refresh, mock_send):
        from src.orchestration.scheduler import run_weekly_watchlist_refresh

        mock_refresh.side_effect = Exception("ウォッチリストエラー")
        run_weekly_watchlist_refresh()  # 例外が外に出ないこと


class TestRunWeeklyOptimization(unittest.TestCase):
    """run_weekly_optimization のテスト"""

    @patch("src.reporting.discord.discord_utils.send_optimization_completion")
    @patch("src.backtest.optimizer.run_optimize_batch")
    @patch("src.watchlist.batch_runner.load_target_symbols")
    def test_sends_completion_notification(self, mock_symbols, mock_run, mock_send):
        from src.orchestration.scheduler import run_weekly_optimization

        mock_symbols.return_value = []
        mock_run.return_value = [{"symbol": "7203", "status": "ok"}]
        run_weekly_optimization()
        mock_send.assert_called_once()

    @patch("src.reporting.discord.discord_utils.send_optimization_completion")
    @patch("src.backtest.optimizer.run_optimize_batch")
    @patch("src.watchlist.batch_runner.load_target_symbols")
    def test_exception_does_not_propagate(self, mock_symbols, mock_run, mock_send):
        from src.orchestration.scheduler import run_weekly_optimization

        mock_symbols.return_value = []
        mock_run.side_effect = Exception("最適化エラー")
        run_weekly_optimization()  # 例外が外に出ないこと


class TestRunDailyPaperTradeReport(unittest.TestCase):
    """run_daily_paper_trade_report のテスト"""

    @patch("src.reporting.discord.discord_utils.send_paper_trade_position_report")
    @patch("src.trading.brokers.paper.paper_broker.PaperBroker")
    def test_paper_mode_sends_report(self, mock_broker_cls, mock_send):
        from unittest.mock import patch as _patch

        from src.orchestration.scheduler import run_daily_paper_trade_report

        mock_broker = mock_broker_cls.return_value
        mock_broker.get_positions.return_value = []
        mock_broker.get_pnl_summary.return_value = {"total_pnl": 0.0}
        with _patch.dict("os.environ", {"AUTO_TRADE_MODE": "paper"}):
            run_daily_paper_trade_report()
        mock_send.assert_called()


# ──────────────────────────────────────────────────────────────
# pytest スタイルの追加テスト
# ──────────────────────────────────────────────────────────────


class TestRunDailyPipeline:
    """run_daily_pipeline のテスト"""

    @patch("src.reporting.discord.discord_utils.send_miss_analysis_summary")
    @patch("src.prediction.db.load_top_prediction_misses")
    @patch("src.reporting.discord.discord_utils.send_daily_pipeline_error")
    @patch("src.reporting.discord.discord_utils.send_daily_pipeline_completion")
    @patch("src.orchestration.jobs.daily.run_daily_drift_check")
    @patch("src.prediction.prediction_pipeline.run_accuracy_check")
    @patch("src.prediction.prediction_pipeline.output_top_worst_results")
    @patch("src.prediction.prediction_pipeline.predict_all_unified")
    @patch("src.market_data.pipeline.run_batch_pipeline")
    @patch("src.watchlist.batch_runner.load_target_symbols")
    def test_daily_pipeline_runs_all_steps(
        self,
        mock_symbols,
        mock_data,
        mock_predict,
        mock_output,
        mock_accuracy,
        mock_drift,
        mock_notify,
        mock_error,
        mock_load_misses,
        mock_send_misses,
    ):
        """全ステップが順番に実行されること"""
        from src.orchestration.scheduler import run_daily_pipeline

        mock_symbols.return_value = []
        mock_predict.return_value = []
        mock_output.return_value = ([], [])
        mock_accuracy.return_value = None
        mock_drift.return_value = None
        mock_notify.return_value = True
        mock_load_misses.return_value = pd.DataFrame()

        run_daily_pipeline()

        mock_data.assert_called_once()
        mock_predict.assert_called_once()
        mock_output.assert_called_once()
        mock_notify.assert_called_once()

    @patch("src.reporting.discord.discord_utils.send_miss_analysis_summary")
    @patch("src.prediction.db.load_top_prediction_misses")
    @patch("src.reporting.discord.discord_utils.send_daily_pipeline_error")
    @patch("src.reporting.discord.discord_utils.send_daily_pipeline_completion")
    @patch("src.orchestration.jobs.daily.run_daily_drift_check")
    @patch("src.prediction.prediction_pipeline.run_accuracy_check")
    @patch("src.prediction.prediction_pipeline.output_top_worst_results")
    @patch("src.prediction.prediction_pipeline.predict_all_unified")
    @patch("src.market_data.pipeline.run_batch_pipeline")
    @patch("src.watchlist.batch_runner.load_target_symbols")
    def test_accuracy_check_failure_does_not_stop_pipeline(
        self,
        mock_symbols,
        mock_data,
        mock_predict,
        mock_output,
        mock_accuracy,
        mock_drift,
        mock_notify,
        mock_error,
        mock_load_misses,
        mock_send_misses,
    ):
        """精度チェックが失敗しても後続ステップが実行されること"""
        from src.orchestration.scheduler import run_daily_pipeline

        mock_symbols.return_value = []
        mock_predict.return_value = []
        mock_output.return_value = ([], [])
        mock_accuracy.side_effect = Exception("精度チェックエラー")
        mock_drift.return_value = None
        mock_notify.return_value = True
        mock_load_misses.return_value = pd.DataFrame()

        run_daily_pipeline()  # 例外が外に伝播しないこと

        mock_drift.assert_called_once()
        mock_notify.assert_called_once()

    @patch("src.reporting.discord.discord_utils.send_daily_pipeline_error")
    @patch("src.market_data.pipeline.run_batch_pipeline")
    @patch("src.watchlist.batch_runner.load_target_symbols")
    def test_data_step_failure_propagates(self, mock_symbols, mock_data, mock_error):
        """データ取得ステップが失敗すると例外が伝播すること"""
        import pytest

        from src.orchestration.scheduler import run_daily_pipeline

        mock_symbols.return_value = []
        mock_data.side_effect = RuntimeError("データ取得失敗")
        mock_error.return_value = None

        with pytest.raises(RuntimeError):
            run_daily_pipeline()


class TestRunWeeklyDbMaintenance(unittest.TestCase):
    """run_weekly_db_maintenance のテスト"""

    @patch("src.utils.db.compact.compact_in_place", return_value={})
    @patch("src.reporting.discord.discord_utils.send_db_maintenance_completion")
    @patch("src.utils.db._db_connection")
    @patch("os.path.getsize", return_value=10 * 1024 * 1024)
    def test_sends_completion_on_success(self, mock_size, mock_conn, mock_send, mock_compact):
        from contextlib import contextmanager

        from src.orchestration.scheduler import run_weekly_db_maintenance

        mock_con = MagicMock()

        @contextmanager
        def fake_conn():
            yield mock_con

        mock_conn.side_effect = fake_conn

        run_weekly_db_maintenance()

        mock_con.execute.assert_any_call("CHECKPOINT")
        mock_con.execute.assert_any_call("VACUUM")
        mock_send.assert_called_once()
        call_kwargs = mock_send.call_args.kwargs
        self.assertIsNone(call_kwargs.get("error"))
        self.assertAlmostEqual(call_kwargs["size_before_mb"], 10.0, places=1)

    @patch("src.reporting.discord.discord_utils.send_db_maintenance_completion")
    @patch("src.utils.db._db_connection")
    @patch("os.path.getsize", return_value=10 * 1024 * 1024)
    def test_sends_error_notification_on_failure(self, mock_size, mock_conn, mock_send):
        from src.orchestration.scheduler import run_weekly_db_maintenance

        mock_conn.side_effect = RuntimeError("DB locked")

        run_weekly_db_maintenance()

        mock_send.assert_called_once()
        call_kwargs = mock_send.call_args.kwargs
        self.assertIsNotNone(call_kwargs.get("error"))

    @patch("src.reporting.discord.discord_utils.send_db_maintenance_completion")
    @patch("src.utils.db._db_connection")
    @patch("os.path.getsize", return_value=10 * 1024 * 1024)
    def test_notification_failure_does_not_propagate(self, mock_size, mock_conn, mock_send):
        from contextlib import contextmanager

        from src.orchestration.scheduler import run_weekly_db_maintenance

        mock_con = MagicMock()

        @contextmanager
        def fake_conn():
            yield mock_con

        mock_conn.side_effect = fake_conn
        mock_send.side_effect = Exception("Webhook 失敗")

        run_weekly_db_maintenance()  # 例外が外に出ないこと


class TestRunWeeklyTraining:
    """run_weekly_training のテスト"""

    @patch("src.reporting.discord.discord_utils.send_weekly_training_completion")
    @patch("src.reporting.discord.discord_utils.send_drift_alert")
    @patch("src.prediction.prediction_pipeline.run_accuracy_check")
    @patch("src.prediction.unified_model_pipeline.train_unified_model")
    def test_weekly_training_calls_train_and_notify(
        self, mock_train, mock_accuracy, mock_drift_alert, mock_notify
    ):
        """学習と通知が実行されること"""
        from src.orchestration.scheduler import run_weekly_training

        mock_train.return_value = None
        mock_accuracy.return_value = None
        mock_drift_alert.return_value = None
        mock_notify.return_value = True

        run_weekly_training()  # 例外が発生しないこと

        assert mock_train.call_count >= 1
        mock_notify.assert_called_once()


class TestRunWeeklyShadowEvaluation(unittest.TestCase):
    """run_weekly_shadow_evaluation のテスト"""

    @patch("src.reporting.discord.discord_utils.send_shadow_evaluation_notification")
    @patch("src.prediction.shadow_evaluation.evaluate_shadow_models")
    def test_sends_notification_when_no_winner(self, mock_evaluate, mock_notify):
        """challenger_wins=False のとき通知が送信されること"""
        from src.orchestration.scheduler import run_weekly_shadow_evaluation

        mock_evaluate.return_value = {
            "production_hit_rate": 0.55,
            "production_sharpe": 0.3,
            "challenger_hit_rate": 0.50,
            "challenger_sharpe": 0.2,
            "challenger_wins": False,
            "evaluated_at": "20260516_120000",
            "n_production": 10,
            "n_challenger": 10,
        }
        run_weekly_shadow_evaluation()
        mock_notify.assert_called_once()
        call_result = mock_notify.call_args.args[0]
        self.assertFalse(call_result["challenger_wins"])

    @patch("src.reporting.discord.discord_utils.send_shadow_evaluation_notification")
    @patch("src.prediction.shadow_evaluation.evaluate_shadow_models")
    def test_sends_notification_when_challenger_wins(self, mock_evaluate, mock_notify):
        """challenger_wins=True のとき昇格候補通知が送信されること"""
        from src.orchestration.scheduler import run_weekly_shadow_evaluation

        mock_evaluate.return_value = {
            "production_hit_rate": 0.50,
            "production_sharpe": 0.2,
            "challenger_hit_rate": 0.65,
            "challenger_sharpe": 0.5,
            "challenger_wins": True,
            "evaluated_at": "20260516_120000",
            "n_production": 10,
            "n_challenger": 10,
        }
        run_weekly_shadow_evaluation()
        mock_notify.assert_called_once()
        call_result = mock_notify.call_args.args[0]
        self.assertTrue(call_result["challenger_wins"])

    @patch("src.reporting.discord.discord_utils.send_shadow_evaluation_notification")
    @patch("src.prediction.shadow_evaluation.evaluate_shadow_models")
    def test_evaluation_failure_does_not_propagate(self, mock_evaluate, mock_notify):
        """evaluate_shadow_models が失敗しても例外が外に出ないこと"""
        from src.orchestration.scheduler import run_weekly_shadow_evaluation

        mock_evaluate.side_effect = Exception("DB エラー")
        run_weekly_shadow_evaluation()  # 例外が外に出ないこと
        mock_notify.assert_not_called()

    @patch("src.reporting.discord.discord_utils.send_shadow_evaluation_notification")
    @patch("src.prediction.shadow_evaluation.evaluate_shadow_models")
    def test_notification_failure_does_not_propagate(self, mock_evaluate, mock_notify):
        """通知が失敗しても例外が外に出ないこと"""
        from src.orchestration.scheduler import run_weekly_shadow_evaluation

        mock_evaluate.return_value = {
            "production_hit_rate": None,
            "production_sharpe": None,
            "challenger_hit_rate": None,
            "challenger_sharpe": None,
            "challenger_wins": False,
            "evaluated_at": "20260516_120000",
            "n_production": 0,
            "n_challenger": 0,
        }
        mock_notify.side_effect = Exception("Webhook 失敗")
        run_weekly_shadow_evaluation()  # 例外が外に出ないこと


# ──────────────────────────────────────────────────────────────
# PipelineStage / _handle_stage_error のユニットテスト
# ──────────────────────────────────────────────────────────────


class TestPipelineStageEnum(unittest.TestCase):
    """PipelineStage 列挙型の基本テスト"""

    def test_has_critical_member(self):
        from src.orchestration.types import PipelineStage

        self.assertIn("CRITICAL", PipelineStage.__members__)

    def test_has_non_critical_member(self):
        from src.orchestration.types import PipelineStage

        self.assertIn("NON_CRITICAL", PipelineStage.__members__)

    def test_has_recoverable_member(self):
        from src.orchestration.types import PipelineStage

        self.assertIn("RECOVERABLE", PipelineStage.__members__)


class TestHandleStageError(unittest.TestCase):
    """_handle_stage_error ヘルパーのユニットテスト"""

    def _call(self, stage, notify_fn=None):
        from src.orchestration.scheduler import _handle_stage_error

        exc = ValueError("test error")
        try:
            raise exc
        except ValueError:
            return _handle_stage_error(stage, "テストラベル", exc, notify_fn)

    def test_critical_returns_true(self):
        from src.orchestration.types import PipelineStage

        result = self._call(PipelineStage.CRITICAL)
        self.assertTrue(result)

    def test_non_critical_returns_false(self):
        from src.orchestration.types import PipelineStage

        result = self._call(PipelineStage.NON_CRITICAL)
        self.assertFalse(result)

    def test_recoverable_returns_false(self):
        from src.orchestration.types import PipelineStage

        result = self._call(PipelineStage.RECOVERABLE)
        self.assertFalse(result)

    def test_critical_calls_notify_fn(self):
        from src.orchestration.types import PipelineStage

        notify_fn = MagicMock()
        self._call(PipelineStage.CRITICAL, notify_fn=notify_fn)
        notify_fn.assert_called_once()

    def test_non_critical_does_not_call_notify_fn(self):
        from src.orchestration.types import PipelineStage

        notify_fn = MagicMock()
        self._call(PipelineStage.NON_CRITICAL, notify_fn=notify_fn)
        notify_fn.assert_not_called()

    def test_recoverable_does_not_call_notify_fn(self):
        from src.orchestration.types import PipelineStage

        notify_fn = MagicMock()
        self._call(PipelineStage.RECOVERABLE, notify_fn=notify_fn)
        notify_fn.assert_not_called()

    def test_critical_with_no_notify_fn_returns_true(self):
        from src.orchestration.types import PipelineStage

        result = self._call(PipelineStage.CRITICAL, notify_fn=None)
        self.assertTrue(result)


class TestIsFirstWeekOfMonth(unittest.TestCase):
    """月初週判定（月次オートコンパクションの発火条件）。"""

    def test_days_1_to_7_are_first_week(self):
        from datetime import datetime, timezone

        from src.orchestration.scheduler import _is_first_week_of_month

        for day in range(1, 8):
            now = datetime(2026, 6, day, tzinfo=timezone.utc)
            self.assertTrue(_is_first_week_of_month(now), f"day={day} should be first week")

    def test_day_8_and_later_are_not_first_week(self):
        from datetime import datetime, timezone

        from src.orchestration.scheduler import _is_first_week_of_month

        for day in (8, 15, 28, 30):
            now = datetime(2026, 6, day, tzinfo=timezone.utc)
            self.assertFalse(_is_first_week_of_month(now), f"day={day} should not be first week")


class TestRunNightlyStrategyFactory(unittest.TestCase):
    """run_nightly_strategy_factory のテスト"""

    def _make_batch_result(self, market="jp"):
        from src.backtest.types import FactoryBatchResult

        return FactoryBatchResult(market=market, champion_sharpe=1.0, pbo=0.3)

    @patch("src.backtest.factory.run_factory_batch")
    def test_skips_when_disabled_and_not_forced(self, mock_batch):
        from src.orchestration.scheduler import run_nightly_strategy_factory

        with patch("config.settings.FACTORY_ENABLED", False):
            run_nightly_strategy_factory(force=False)

        mock_batch.assert_not_called()

    @patch("src.reporting.discord.discord_utils.send_factory_completion")
    @patch("src.watchlist.batch_runner.load_target_symbols")
    @patch("src.backtest.factory.run_factory_batch")
    def test_forced_run_filters_symbols_by_market(self, mock_batch, mock_load, mock_send):
        from src.orchestration.scheduler import run_nightly_strategy_factory

        mock_load.return_value = [
            SymbolTask(market="jp", symbol="7203"),
            SymbolTask(market="jp", symbol="9984"),
            SymbolTask(market="us", symbol="AAPL"),
        ]
        mock_batch.return_value = self._make_batch_result("jp")

        with patch("config.settings.FACTORY_ENABLED", False):
            run_nightly_strategy_factory(force=True, market="jp", budget=3, seed=1)

        kwargs = mock_batch.call_args.kwargs
        self.assertEqual(kwargs["market"], "jp")
        self.assertEqual(kwargs["symbols"], ["7203", "9984"])
        self.assertEqual(kwargs["budget"], 3)
        mock_send.assert_called_once()

    @patch("src.reporting.discord.discord_utils.send_factory_completion")
    @patch("src.watchlist.batch_runner.load_target_symbols")
    @patch("src.backtest.factory.run_factory_batch")
    def test_default_market_is_jp_or_us(self, mock_batch, mock_load, mock_send):
        from src.orchestration.scheduler import run_nightly_strategy_factory

        mock_load.return_value = [
            SymbolTask(market="jp", symbol="7203"),
            SymbolTask(market="us", symbol="AAPL"),
        ]
        mock_batch.side_effect = lambda **kw: self._make_batch_result(kw["market"])

        with patch("config.settings.FACTORY_ENABLED", True):
            run_nightly_strategy_factory(force=False)

        self.assertIn(mock_batch.call_args.kwargs["market"], ("jp", "us"))

    @patch("src.reporting.discord.discord_utils.send_factory_completion")
    @patch("src.watchlist.batch_runner.load_target_symbols")
    @patch("src.backtest.factory.run_factory_batch")
    def test_no_symbols_aborts_without_batch(self, mock_batch, mock_load, mock_send):
        from src.orchestration.scheduler import run_nightly_strategy_factory

        mock_load.return_value = [SymbolTask(market="us", symbol="AAPL")]

        run_nightly_strategy_factory(force=True, market="jp")

        mock_batch.assert_not_called()
        mock_send.assert_not_called()

    @patch("src.reporting.discord.discord_utils.send_factory_completion")
    @patch("src.watchlist.batch_runner.load_target_symbols")
    @patch("src.backtest.factory.run_factory_batch")
    def test_batch_error_does_not_raise_and_skips_notification(
        self, mock_batch, mock_load, mock_send
    ):
        from src.orchestration.scheduler import run_nightly_strategy_factory

        mock_load.return_value = [SymbolTask(market="jp", symbol="7203")]
        mock_batch.side_effect = RuntimeError("evaluation failed")

        run_nightly_strategy_factory(force=True, market="jp")  # 例外が外に漏れないこと

        mock_send.assert_not_called()
