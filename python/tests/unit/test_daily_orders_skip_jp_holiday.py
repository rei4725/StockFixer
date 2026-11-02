"""東証の休場日には 8:50 の自動発注と 16:30 のルールの発注を出さないテスト（#1185）。

休場日の注文は次の営業日の寄りで約定するため、休場明けの 8:50 の分と合わせて
同じ足の情報で2回分の新規買いが建ち、1回の買いの上限が効かなくなる。
"""

import unittest
from contextlib import ExitStack
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from src.rule_engine.types import RuleSignalBatch

JST = ZoneInfo("Asia/Tokyo")
HOLIDAY = datetime(2026, 11, 3, 8, 50, tzinfo=JST)  # 文化の日（火）
BUSINESS_DAY = datetime(2026, 11, 4, 8, 50, tzinfo=JST)


class TestAutoOrderSkipsJpHoliday(unittest.TestCase):
    def _run(self, now: datetime) -> tuple[MagicMock, MagicMock, MagicMock]:
        from src.orchestration.jobs import common, daily

        broker = MagicMock()
        stats = {"buy_orders": 1, "sell_orders": 0, "errors": 0}
        with ExitStack() as stack:
            stack.enter_context(patch.dict("os.environ", {"AUTO_TRADE_MODE": "paper"}))
            stack.enter_context(patch.object(common, "now_jst", return_value=now))

            def run_orders(**kwargs):
                kwargs["broker"].send_order("7203", "BUY", 100)
                return stats

            run_orders_mock = stack.enter_context(
                patch("src.trading.execution.run_daily_orders", side_effect=run_orders)
            )
            stack.enter_context(
                patch(
                    "src.trading.brokers.account.factory.make_order_broker",
                    return_value=broker,
                )
            )
            for target in (
                "src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter",
                "src.infrastructure.discord_notification_adapter.DiscordNotificationAdapter",
                "src.infrastructure.persistence.order_run_repository.PostgresOrderRunSink",
                "src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink",
                "src.infrastructure.persistence.prediction_result_repository."
                "PostgresPredictionResultRepository",
                "src.reporting.discord.discord_utils.send_daily_order_completion",
            ):
                stack.enter_context(patch(target))
            stack.enter_context(patch.object(daily, "record_loss_limit_streak"))
            horizon_exit = stack.enter_context(patch.object(daily, "run_horizon_exit_check"))
            daily.run_daily_auto_order()
        return run_orders_mock, broker, horizon_exit

    def test_holiday_places_no_orders(self):
        with self.assertLogs("src.orchestration.jobs.common", level="INFO") as logs:
            run_orders, broker, horizon_exit = self._run(HOLIDAY)
        run_orders.assert_not_called()
        broker.send_order.assert_not_called()
        horizon_exit.assert_not_called()
        self.assertTrue(any("休場日" in line for line in logs.output))

    def test_business_day_places_orders(self):
        run_orders, broker, horizon_exit = self._run(BUSINESS_DAY)
        run_orders.assert_called_once()
        broker.send_order.assert_called_once()
        horizon_exit.assert_called_once_with()


class TestRuleSignalsSkipJpHoliday(unittest.TestCase):
    def _run(self, now: datetime, market: str) -> tuple[MagicMock, MagicMock, MagicMock]:
        from src.orchestration.jobs import common, daily

        broker = MagicMock()
        batch = RuleSignalBatch(
            signals=[{"symbol": "7203", "rule": "rsi_contrarian", "signal": 1, "price": 100.0}],
            unavailable=[],
            total=1,
        )

        def execute(**kwargs):
            kwargs["broker"].send_order("7203", "BUY", 100)
            return {"buy_orders": 1, "sell_orders": 0, "errors": 0}

        with ExitStack() as stack:
            stack.enter_context(patch.dict("os.environ", {"RULE_EVAL_MARKET": market}))
            stack.enter_context(patch.object(common, "now_jst", return_value=now))
            generate = stack.enter_context(
                patch("src.rule_engine.pipeline.generate_rule_signals", return_value=batch)
            )
            execute_mock = stack.enter_context(
                patch("src.trading.rule_execution.execute_rule_paper_trades", side_effect=execute)
            )
            stack.enter_context(
                patch(
                    "src.trading.brokers.account.factory.make_order_broker",
                    return_value=broker,
                )
            )
            for target in (
                "src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter",
                "src.infrastructure.discord_notification_adapter.DiscordNotificationAdapter",
                "src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink",
                "src.reporting.discord.discord_utils.send_rule_daily_signals",
            ):
                stack.enter_context(patch(target))
            daily.run_daily_rule_signals()
        return generate, execute_mock, broker

    def test_jp_holiday_places_no_orders(self):
        with self.assertLogs("src.orchestration.jobs.common", level="INFO") as logs:
            generate, execute, broker = self._run(HOLIDAY, "jp")
        generate.assert_not_called()
        execute.assert_not_called()
        broker.send_order.assert_not_called()
        self.assertTrue(any("休場日" in line for line in logs.output))

    def test_jp_business_day_places_orders(self):
        _, execute, broker = self._run(BUSINESS_DAY, "jp")
        execute.assert_called_once()
        broker.send_order.assert_called_once()

    def test_non_jp_market_places_orders_on_jp_holiday(self):
        _, execute, broker = self._run(HOLIDAY, "us")
        execute.assert_called_once()
        broker.send_order.assert_called_once()


if __name__ == "__main__":
    unittest.main()
