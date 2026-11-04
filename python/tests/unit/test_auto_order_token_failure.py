"""8:50 の発注でトークンを取れない日を、errors とジョブの失敗と知らせで見せるテスト（#1523）。

その日は SL/TP も含めて全部止まるので、成功の完了に見せず、ERROR の知らせを出して失敗で終える。
トークンを取れた日（紙の日）は、errors も知らせも増えない。
その朝は日ごとの記録にも失敗を書いてから返す（#1533）。
"""

import json
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from src.domain.ports import AlertLevel
from src.infrastructure.in_memory import (
    InMemoryNotificationAdapter,
    InMemoryOrderRunSink,
    InMemoryPredictionRepository,
    InMemoryTradeDiffSink,
)
from src.trading.brokers.base import BrokerBase, BrokerError


def _token_failing_broker() -> MagicMock:
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_token.side_effect = BrokerError("token error")
    return broker


class TestRunDailyOrdersCountsTokenFailure(unittest.TestCase):
    def _run(self, broker, notifier):
        from src.trading.execution import run_daily_orders

        return run_daily_orders(
            broker,
            order_run_sink=InMemoryOrderRunSink(),
            trade_diff_sink=InMemoryTradeDiffSink(),
            prediction_repo=InMemoryPredictionRepository(),
            market="jp",
            mode="paper",
            notifier=notifier,
        )

    def test_token_failure_counts_one_error(self):
        notifier = InMemoryNotificationAdapter()

        with self.assertLogs(level="ERROR"):
            stats = self._run(_token_failing_broker(), notifier)

        self.assertEqual(stats["errors"], 1)
        self.assertEqual(stats["buy_orders"], 0)
        self.assertEqual(stats["sell_orders"], 0)
        self.assertEqual([a["level"] for a in notifier.sent_alerts], [AlertLevel.ERROR])

    def test_token_failure_without_notifier_still_counts(self):
        with self.assertLogs(level="ERROR"):
            stats = self._run(_token_failing_broker(), None)

        self.assertEqual(stats["errors"], 1)

    def test_token_ok_day_adds_no_error_and_no_alert(self):
        broker = MagicMock(spec=BrokerBase)
        broker.broker_name = "paper"
        notifier = InMemoryNotificationAdapter()

        with (
            patch("src.trading.execution.runner.RiskManager"),
            patch("src.trading.execution.runner._evaluate_entry_gate") as gate,
            patch("src.trading.execution.runner._sl_tp_exits_only") as exits_only,
        ):
            gate.return_value = (MagicMock(), None)
            stats = self._run(broker, notifier)

        exits_only.assert_called_once()
        self.assertEqual(stats["errors"], 0)
        self.assertEqual(notifier.sent_alerts, [])


class TestTokenFailureOrderRecord(unittest.TestCase):
    """トークンを取れない朝も results/orders/<JST の日付>.json に失敗が残る（#1533）。"""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dest = Path(tmp.name) / "orders" / "2026-11-04.json"
        for target, value in (
            ("get_results_dir", tmp.name),
            ("now_jst", datetime(2026, 11, 4, 8, 50, tzinfo=ZoneInfo("Asia/Tokyo"))),
        ):
            started = patch(f"src.trading.execution.gate_record.{target}", return_value=value)
            started.start()
            self.addCleanup(started.stop)
        self.broker = _token_failing_broker()
        self.notifier = InMemoryNotificationAdapter()

    def _run(self):
        from src.trading.execution import run_daily_orders

        return run_daily_orders(
            self.broker,
            order_run_sink=InMemoryOrderRunSink(),
            trade_diff_sink=InMemoryTradeDiffSink(),
            prediction_repo=InMemoryPredictionRepository(),
            market="jp",
            mode="paper",
            notifier=self.notifier,
        )

    def _assert_unchanged_outcome(self, stats):
        self.assertEqual(stats["errors"], 1)
        self.assertEqual(stats["buy_orders"], 0)
        self.assertEqual(stats["sell_orders"], 0)
        self.broker.send_order.assert_not_called()
        self.assertEqual([a["level"] for a in self.notifier.sent_alerts], [AlertLevel.ERROR])

    def test_token_failure_is_written_to_the_day_record(self):
        with self.assertLogs(level="ERROR"):
            stats = self._run()

        self._assert_unchanged_outcome(stats)
        record = json.loads(self.dest.read_text(encoding="utf-8"))
        self.assertEqual(record["date"], "2026-11-04")
        self.assertEqual(record["mode"], "paper")
        self.assertIn("token error", record["token_error"])
        self.assertIsNone(record["entry_error"])
        self.assertEqual(record["errors"], 1)
        self.assertEqual(record["buy_orders"], 0)
        self.assertEqual(record["sell_orders"], 0)

    def test_write_failure_keeps_the_same_outcome(self):
        with (
            patch(
                "src.trading.execution.gate_record._write_record",
                side_effect=OSError("disk full"),
            ),
            self.assertLogs("src.trading.execution.gate_record", level="ERROR"),
        ):
            stats = self._run()

        self._assert_unchanged_outcome(stats)
        self.assertFalse(self.dest.exists())

    def test_token_ok_day_records_null(self):
        broker = MagicMock(spec=BrokerBase)
        broker.broker_name = "paper"
        self.broker = broker

        with (
            patch("src.trading.execution.runner.RiskManager"),
            patch("src.trading.execution.runner._evaluate_entry_gate") as gate,
            patch("src.trading.execution.runner._sl_tp_exits_only"),
        ):
            gate.return_value = (MagicMock(), None)
            self._run()

        record = json.loads(self.dest.read_text(encoding="utf-8"))
        self.assertIsNone(record["token_error"])


class TestDailyAutoOrderTokenFailure(unittest.TestCase):
    def setUp(self):
        from src.orchestration.jobs import daily

        self.daily = daily
        self.broker = _token_failing_broker()

        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            patch(
                "src.trading.brokers.account.factory.make_order_broker",
                return_value=self.broker,
            )
        )
        self.notifier_cls = stack.enter_context(
            patch("src.infrastructure.discord_notification_adapter.DiscordNotificationAdapter")
        )
        for target in (
            "src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter",
            "src.infrastructure.persistence.order_run_repository.PostgresOrderRunSink",
            "src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink",
            "src.infrastructure.persistence.prediction_result_repository."
            "PostgresPredictionResultRepository",
        ):
            stack.enter_context(patch(target))
        stack.enter_context(patch.object(daily, "_skip_if_jp_market_closed", return_value=False))
        stack.enter_context(patch.object(daily, "record_loss_limit_streak"))
        stack.enter_context(patch.object(daily, "notify_stale_prediction"))
        stack.enter_context(patch.object(daily, "read_prediction_saved_day", return_value=None))
        self.completion = stack.enter_context(
            patch("src.reporting.discord.discord_utils.send_daily_order_completion")
        )
        stack.enter_context(patch.object(daily, "run_horizon_exit_check"))

    def test_token_failure_day_ends_job_with_exception_and_alert(self):
        with self.assertLogs(level="ERROR"):
            with self.assertRaises(RuntimeError):
                self.daily.run_daily_auto_order()

        self.broker.get_token.assert_called_once_with()
        self.broker.send_order.assert_not_called()
        send_alert = self.notifier_cls.return_value.send_alert
        send_alert.assert_called_once()
        self.assertEqual(send_alert.call_args.kwargs["level"], AlertLevel.ERROR)
        self.assertEqual(self.completion.call_args.kwargs["errors"], 1)

    def test_passes_notifier_to_run_daily_orders(self):
        stats = {"buy_orders": 0, "sell_orders": 0, "errors": 0}

        with patch("src.trading.execution.run_daily_orders", return_value=stats) as run_orders:
            self.daily.run_daily_auto_order()

        self.assertIs(run_orders.call_args.kwargs["notifier"], self.notifier_cls.return_value)
        self.notifier_cls.return_value.send_alert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
