"""8:50 の自動発注で、注文ごとの失敗をジョブの失敗と知らせのエラーの件数にするテスト（#1074）。

run_scheduler の daily_auto_order は max_retries: 0 のため、ジョブを最後に失敗にしても
run_daily_orders は同じ回の中で呼び直されない（呼ばれるのが1回であることも確かめる）。
"""

import unittest
from contextlib import ExitStack
from unittest.mock import ANY, MagicMock, patch

from src.trading.brokers.base import OrderSide


class TestAutoOrderErrorsFailJob(unittest.TestCase):
    def setUp(self):
        from src.orchestration.jobs import daily

        self.daily = daily
        self.stats = {
            "buy_orders": 0,
            "sell_orders": 0,
            "errors": 0,
            "total_turnover": 0.0,
            "correlation_blocked": True,
        }
        self.broker = MagicMock()
        self.broker.get_positions.return_value = [
            {"symbol": "7203", "qty": 100, "avg_price": 1000.0, "current_price": 900.0},
            {"symbol": "9984", "qty": 100, "avg_price": 1000.0, "current_price": 900.0},
        ]

        def send_order(symbol, side, qty, **_):
            if symbol == "7203":
                raise RuntimeError("broker down")
            return MagicMock()

        self.broker.send_order.side_effect = send_order

        stack = ExitStack()
        self.addCleanup(stack.close)
        self.run_orders = stack.enter_context(
            patch("src.trading.execution.run_daily_orders", side_effect=self._run_sl_tp)
        )
        stack.enter_context(
            patch(
                "src.trading.brokers.account.factory.make_order_broker",
                return_value=self.broker,
            )
        )
        for target in (
            "src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter",
            "src.infrastructure.discord_notification_adapter.DiscordNotificationAdapter",
            "src.infrastructure.persistence.order_run_repository.PostgresOrderRunSink",
            "src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink",
            "src.infrastructure.persistence.prediction_result_repository."
            "PostgresPredictionResultRepository",
            "src.trading.execution.sl_tp._record_order",
        ):
            stack.enter_context(patch(target))
        risk = stack.enter_context(patch("src.trading.execution.sl_tp.RiskManager"))
        risk.return_value.check_sl_tp.return_value = (True, False, "SL")
        stack.enter_context(patch.object(daily, "record_loss_limit_streak"))
        self.completion = stack.enter_context(
            patch("src.reporting.discord.discord_utils.send_daily_order_completion")
        )
        self.correlation_alert = stack.enter_context(
            patch("src.reporting.discord.discord_utils.send_correlation_alert")
        )
        self.horizon_exit = stack.enter_context(patch.object(daily, "run_horizon_exit_check"))

    def _run_sl_tp(self, **kwargs):
        from src.trading.execution.sl_tp import _check_sl_tp_exits

        _check_sl_tp_exits(
            kwargs["broker"], "jp", "paper", None, self.stats, kwargs["trade_diff_sink"]
        )
        return self.stats

    def test_sl_tp_sell_error_fails_job_after_other_orders(self):
        with self.assertLogs(level="ERROR"):
            with self.assertRaises(RuntimeError):
                self.daily.run_daily_auto_order()

        self.broker.send_order.assert_any_call(
            "9984", OrderSide.SELL, 100, price=0.0, order_type=ANY
        )
        self.assertEqual(self.stats["sell_orders"], 1)
        self.assertEqual(self.completion.call_args.kwargs["errors"], 1)
        self.horizon_exit.assert_called_once_with()
        self.correlation_alert.assert_called_once()
        self.run_orders.assert_called_once()

    def test_no_errors_finishes_normally(self):
        self.broker.send_order.side_effect = None

        self.daily.run_daily_auto_order()

        self.assertEqual(self.stats["sell_orders"], 2)
        self.assertEqual(self.completion.call_args.kwargs["errors"], 0)
        self.correlation_alert.assert_called_once()
        self.run_orders.assert_called_once()


if __name__ == "__main__":
    unittest.main()
