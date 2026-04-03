"""ユニットテスト: scheduler_pipeline の自動発注連携"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.services.scheduler_pipeline import run_daily_auto_order


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


if __name__ == "__main__":
    unittest.main()
