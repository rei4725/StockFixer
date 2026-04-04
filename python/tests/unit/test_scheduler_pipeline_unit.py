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
