"""自動発注のジョブがホライズンの期限の決済の失敗を握りつぶさないことのテスト（#886）。"""

import unittest
from contextlib import ExitStack
from unittest.mock import patch


class TestAutoOrderHorizonExitFailure(unittest.TestCase):
    def setUp(self):
        from src.orchestration.jobs import daily

        self.daily = daily
        self.stats = {
            "buy_orders": 1,
            "sell_orders": 0,
            "correlation_blocked": False,
        }
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            patch("src.trading.execution.run_daily_orders", side_effect=lambda **_: self.stats)
        )
        for target in (
            "src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter",
            "src.infrastructure.discord_notification_adapter.DiscordNotificationAdapter",
            "src.trading.brokers.account.factory.make_order_broker",
            "src.infrastructure.persistence.order_run_repository.PostgresOrderRunSink",
            "src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink",
            "src.infrastructure.persistence.prediction_result_repository."
            "PostgresPredictionResultRepository",
        ):
            stack.enter_context(patch(target))
        stack.enter_context(patch.object(daily, "record_loss_limit_streak"))
        self.completion = stack.enter_context(
            patch("src.reporting.discord.discord_utils.send_daily_order_completion")
        )
        self.correlation_alert = stack.enter_context(
            patch("src.reporting.discord.discord_utils.send_correlation_alert")
        )
        self.horizon_exit = stack.enter_context(patch.object(daily, "run_horizon_exit_check"))

    def test_horizon_exit_failure_fails_job(self):
        self.horizon_exit.side_effect = RuntimeError("db down")

        with self.assertLogs("src.orchestration.jobs.daily", level="ERROR"):
            with self.assertRaises(RuntimeError):
                self.daily.run_daily_auto_order()

        self.completion.assert_called_once()
        self.correlation_alert.assert_not_called()

    def test_horizon_exit_failure_still_sends_correlation_alert(self):
        self.horizon_exit.side_effect = RuntimeError("db down")
        self.stats["correlation_blocked"] = True

        with self.assertLogs("src.orchestration.jobs.daily", level="ERROR"):
            with self.assertRaises(RuntimeError):
                self.daily.run_daily_auto_order()

        self.correlation_alert.assert_called_once()

    def test_horizon_exit_success_finishes_normally(self):
        self.stats["correlation_blocked"] = True

        self.daily.run_daily_auto_order()

        self.horizon_exit.assert_called_once_with()
        self.completion.assert_called_once()
        self.correlation_alert.assert_called_once()


if __name__ == "__main__":
    unittest.main()
