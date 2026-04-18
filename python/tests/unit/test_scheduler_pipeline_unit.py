"""ユニットテスト: scheduler_pipeline の自動発注連携"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from src.domain.types import SymbolTask
from src.services.scheduler_pipeline import run_daily_auto_order, run_daily_drift_check


class TestRunDailyAutoOrder(unittest.TestCase):
    @patch("src.api.discord_utils.send_daily_order_completion")
    @patch("src.services.order_execution_pipeline.run_daily_orders")
    @patch("src.brokers.paper.paper_broker.PaperBroker")
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

        mock_run_daily_orders.assert_called_once_with(broker=broker, market="jp", mode="paper")
        mock_send_completion.assert_called_once_with(
            buy_orders=0,
            sell_orders=0,
            mode="paper",
            trading_stopped=True,
            stop_reason="日次損失上限に到達",
            daily_loss=25_000.0,
            daily_loss_limit=20_000.0,
        )

    @patch("src.api.discord_utils.send_daily_order_completion")
    @patch("src.services.order_execution_pipeline.run_daily_orders")
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
            patch.dict(sys.modules, {"src.brokers.kabu.kabu_client": fake_kabu_module}),
        ):
            run_daily_auto_order()

        mock_run_daily_orders.assert_called_once_with(broker=broker, market="jp", mode="live")
        mock_send_completion.assert_called_once()


class TestRunDailyDriftCheck(unittest.TestCase):
    @patch("src.services.model_training_pipeline.train_models_for_symbol_task")
    @patch("src.services.batch_runner.load_target_symbols")
    @patch("src.api.discord_utils.send_drift_retrain_notification")
    @patch("src.utils.db.load_drift_summary")
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

    @patch("src.services.model_training_pipeline.train_models_for_symbol_task")
    @patch("src.services.batch_runner.load_target_symbols")
    @patch("src.api.discord_utils.send_drift_retrain_notification")
    @patch("src.utils.db.load_drift_summary")
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

    @patch("src.services.model_training_pipeline.train_models_for_symbol_task")
    @patch("src.services.batch_runner.load_target_symbols")
    @patch("src.api.discord_utils.send_drift_retrain_notification")
    @patch("src.utils.db.load_drift_summary")
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


class TestRunWeeklyReport(unittest.TestCase):
    """run_weekly_report のテスト"""

    @patch("src.api.discord_utils.send_weekly_report")
    @patch("src.utils.db.load_paper_real_diff_summary")
    @patch("src.utils.db.load_drift_summary")
    def test_calls_send_weekly_report(self, mock_drift, mock_diff, mock_send):
        import pandas as pd

        from src.services.scheduler_pipeline import run_weekly_report

        mock_drift.return_value = pd.DataFrame({"symbol": ["7203"], "direction_accuracy": [0.6]})
        mock_diff.return_value = pd.DataFrame({"symbol": ["7203"], "diff": [0.01]})
        run_weekly_report()
        mock_send.assert_called_once()

    @patch("src.api.discord_utils.send_weekly_report")
    @patch("src.utils.db.load_paper_real_diff_summary")
    @patch("src.utils.db.load_drift_summary")
    def test_exception_does_not_propagate(self, mock_drift, mock_diff, mock_send):
        from src.services.scheduler_pipeline import run_weekly_report

        mock_drift.side_effect = Exception("DB エラー")
        run_weekly_report()  # 例外が外に出ないこと


class TestRunDailySettle(unittest.TestCase):
    """run_daily_settle_orders のテスト"""

    @patch("src.api.discord_utils.send_daily_settle_completion")
    @patch("src.brokers.paper.paper_broker.PaperBroker")
    def test_paper_mode_calls_settle(self, mock_broker_cls, mock_send):
        from unittest.mock import patch as _patch

        from src.services.scheduler_pipeline import run_daily_settle_orders

        mock_broker = mock_broker_cls.return_value
        mock_broker.settle_pending_orders.return_value = [{"symbol": "7203", "qty": 10}]
        with _patch.dict("os.environ", {"AUTO_TRADE_MODE": "paper"}):
            run_daily_settle_orders()
        mock_broker.settle_pending_orders.assert_called()

    @patch("src.api.discord_utils.send_daily_settle_completion")
    def test_live_mode_is_skipped(self, mock_send):
        from unittest.mock import patch as _patch

        from src.services.scheduler_pipeline import run_daily_settle_orders

        with _patch.dict("os.environ", {"AUTO_TRADE_MODE": "live"}):
            run_daily_settle_orders()
        mock_send.assert_not_called()


class TestRunWeeklyWalkForwardReport(unittest.TestCase):
    """run_weekly_walk_forward_report のテスト"""

    @patch("src.api.discord_utils.send_walk_forward_report_completion")
    @patch("src.services.walk_forward_report_pipeline.run_walk_forward_comparison_report")
    def test_sends_completion_notification(self, mock_run, mock_send):
        from src.services.scheduler_pipeline import run_weekly_walk_forward_report

        mock_run.return_value = {"success": 5, "failed": 1, "total": 6}
        run_weekly_walk_forward_report()
        mock_send.assert_called_once()

    @patch("src.api.discord_utils.send_walk_forward_report_completion")
    @patch("src.services.walk_forward_report_pipeline.run_walk_forward_comparison_report")
    def test_exception_does_not_propagate(self, mock_run, mock_send):
        from src.services.scheduler_pipeline import run_weekly_walk_forward_report

        mock_run.side_effect = Exception("レポートエラー")
        run_weekly_walk_forward_report()  # 例外が外に出ないこと


class TestRunWeeklyWatchlistRefresh(unittest.TestCase):
    """run_weekly_watchlist_refresh のテスト"""

    @patch("src.api.discord_utils.send_watchlist_update_report")
    @patch("src.services.watchlist_manager.run_watchlist_refresh")
    def test_sends_update_report(self, mock_refresh, mock_send):
        from src.services.scheduler_pipeline import run_weekly_watchlist_refresh

        mock_refresh.return_value = []
        run_weekly_watchlist_refresh()
        mock_send.assert_called_once()

    @patch("src.api.discord_utils.send_watchlist_update_report")
    @patch("src.services.watchlist_manager.run_watchlist_refresh")
    def test_exception_does_not_propagate(self, mock_refresh, mock_send):
        from src.services.scheduler_pipeline import run_weekly_watchlist_refresh

        mock_refresh.side_effect = Exception("ウォッチリストエラー")
        run_weekly_watchlist_refresh()  # 例外が外に出ないこと


class TestRunWeeklyOptimization(unittest.TestCase):
    """run_weekly_optimization のテスト"""

    @patch("src.api.discord_utils.send_optimization_completion")
    @patch("src.services.backtest_optimize_pipeline.run_optimize_batch")
    def test_sends_completion_notification(self, mock_run, mock_send):
        from src.services.scheduler_pipeline import run_weekly_optimization

        mock_run.return_value = [{"symbol": "7203", "status": "ok"}]
        run_weekly_optimization()
        mock_send.assert_called_once()

    @patch("src.api.discord_utils.send_optimization_completion")
    @patch("src.services.backtest_optimize_pipeline.run_optimize_batch")
    def test_exception_does_not_propagate(self, mock_run, mock_send):
        from src.services.scheduler_pipeline import run_weekly_optimization

        mock_run.side_effect = Exception("最適化エラー")
        run_weekly_optimization()  # 例外が外に出ないこと


class TestRunDailyPaperTradeReport(unittest.TestCase):
    """run_daily_paper_trade_report のテスト"""

    @patch("src.api.discord_utils.send_paper_trade_position_report")
    @patch("src.brokers.paper.paper_broker.PaperBroker")
    def test_paper_mode_sends_report(self, mock_broker_cls, mock_send):
        from unittest.mock import patch as _patch

        from src.services.scheduler_pipeline import run_daily_paper_trade_report

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

    @patch("src.api.discord_utils.send_daily_pipeline_error")
    @patch("src.api.discord_utils.send_daily_pipeline_completion")
    @patch("src.services.scheduler_pipeline.run_daily_drift_check")
    @patch("src.services.prediction_pipeline.run_accuracy_check")
    @patch("src.services.prediction_pipeline.output_top_worst_results")
    @patch("src.services.prediction_pipeline.predict_all_unified")
    @patch("src.services.data_pipeline.run_data_batch")
    def test_daily_pipeline_runs_all_steps(
        self,
        mock_data,
        mock_predict,
        mock_output,
        mock_accuracy,
        mock_drift,
        mock_notify,
        mock_error,
    ):
        """全ステップが順番に実行されること"""
        from src.services.scheduler_pipeline import run_daily_pipeline

        mock_predict.return_value = []
        mock_output.return_value = ([], [])
        mock_accuracy.return_value = None
        mock_drift.return_value = None
        mock_notify.return_value = True

        run_daily_pipeline()

        mock_data.assert_called_once()
        mock_predict.assert_called_once()
        mock_output.assert_called_once()
        mock_notify.assert_called_once()

    @patch("src.api.discord_utils.send_daily_pipeline_error")
    @patch("src.api.discord_utils.send_daily_pipeline_completion")
    @patch("src.services.scheduler_pipeline.run_daily_drift_check")
    @patch("src.services.prediction_pipeline.run_accuracy_check")
    @patch("src.services.prediction_pipeline.output_top_worst_results")
    @patch("src.services.prediction_pipeline.predict_all_unified")
    @patch("src.services.data_pipeline.run_data_batch")
    def test_accuracy_check_failure_does_not_stop_pipeline(
        self,
        mock_data,
        mock_predict,
        mock_output,
        mock_accuracy,
        mock_drift,
        mock_notify,
        mock_error,
    ):
        """精度チェックが失敗しても後続ステップが実行されること"""
        import pytest

        from src.services.scheduler_pipeline import run_daily_pipeline

        mock_predict.return_value = []
        mock_output.return_value = ([], [])
        mock_accuracy.side_effect = Exception("精度チェックエラー")
        mock_drift.return_value = None
        mock_notify.return_value = True

        run_daily_pipeline()  # 例外が外に伝播しないこと

        mock_drift.assert_called_once()
        mock_notify.assert_called_once()

    @patch("src.api.discord_utils.send_daily_pipeline_error")
    @patch("src.services.data_pipeline.run_data_batch")
    def test_data_step_failure_propagates(self, mock_data, mock_error):
        """データ取得ステップが失敗すると例外が伝播すること"""
        import pytest

        from src.services.scheduler_pipeline import run_daily_pipeline

        mock_data.side_effect = RuntimeError("データ取得失敗")
        mock_error.return_value = None

        with pytest.raises(Exception):
            run_daily_pipeline()


class TestRunWeeklyTraining:
    """run_weekly_training のテスト"""

    @patch("src.api.discord_utils.send_weekly_training_completion")
    @patch("src.api.discord_utils.send_drift_alert")
    @patch("src.services.prediction_pipeline.run_accuracy_check")
    @patch("src.services.unified_model_pipeline.train_unified_model")
    def test_weekly_training_calls_train_and_notify(
        self, mock_train, mock_accuracy, mock_drift_alert, mock_notify
    ):
        """学習と通知が実行されること"""
        from src.services.scheduler_pipeline import run_weekly_training

        mock_train.return_value = None
        mock_accuracy.return_value = None
        mock_drift_alert.return_value = None
        mock_notify.return_value = True

        run_weekly_training()  # 例外が発生しないこと

        assert mock_train.call_count >= 1
        mock_notify.assert_called_once()
