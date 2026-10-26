"""ユニットテスト: 日次ルールシグナルのジョブが取れなかった銘柄を数える（#822）"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.rule_engine.types import RuleSignalBatch, UnavailableRuleSymbol


def _batch(n_ok: int, n_unavailable: int) -> RuleSignalBatch:
    signals = [
        {"symbol": f"OK{i}", "rule": "rsi_contrarian", "signal": -1, "price": 100.0}
        for i in range(n_ok)
    ]
    unavailable = [
        UnavailableRuleSymbol(symbol=f"NG{i}", rule="rsi_contrarian", reason="データ不足")
        for i in range(n_unavailable)
    ]
    return RuleSignalBatch(signals=signals, unavailable=unavailable, total=n_ok + n_unavailable)


class TestRunDailyRuleSignalsUnavailable(unittest.TestCase):
    def _run(self, batch: RuleSignalBatch) -> MagicMock:
        from src.orchestration.jobs import daily

        execute = MagicMock(return_value={"buy_orders": 0, "sell_orders": 1})
        with (
            patch("src.rule_engine.pipeline.generate_rule_signals", return_value=batch),
            patch("src.trading.rule_execution.execute_rule_paper_trades", execute),
            patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter"),
            patch("src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink"),
            patch("src.reporting.discord.discord_utils.send_rule_daily_signals"),
        ):
            daily.run_daily_rule_signals()
        return execute

    def test_raises_when_more_than_half_unavailable(self):
        with self.assertLogs("src.orchestration.jobs.daily", level="ERROR") as logs:
            with self.assertRaises(RuntimeError):
                self._run(_batch(n_ok=1, n_unavailable=2))
        self.assertIn("日次ルールシグナル失敗", "\n".join(logs.output))

    def test_places_held_sells_but_no_new_buys_when_more_than_half_unavailable(self):
        """半分を超えた日も保有の SELL だけは出し、新規の BUY は出さずに失敗にする（#1191）。"""
        from src.orchestration.jobs import daily
        from src.trading.brokers.base import BrokerBase, OrderSide, OrderType

        batch = RuleSignalBatch(
            signals=[
                {"symbol": "1000", "rule": "rsi_contrarian", "signal": -1, "price": 100.0},
                {"symbol": "2000", "rule": "rsi_contrarian", "signal": 1, "price": 100.0},
            ],
            unavailable=[
                UnavailableRuleSymbol(symbol=f"NG{i}", rule="rsi_contrarian", reason="データ不足")
                for i in range(3)
            ],
            total=5,
        )
        broker = MagicMock(spec=BrokerBase)
        broker.broker_name = "mock"
        broker.get_positions.return_value = [
            {"symbol": "1000", "qty": 100, "avg_price": 100.0, "current_price": 100.0}
        ]
        risk = MagicMock()
        risk.evaluate_trading_gate.return_value = MagicMock(is_allowed=True, reason=None)
        risk.calc_position_size.return_value = 100
        with (
            patch("src.rule_engine.pipeline.generate_rule_signals", return_value=batch),
            patch("src.trading.brokers.account.factory.make_order_broker", return_value=broker),
            patch("src.trading.rule_execution.RiskManager", return_value=risk),
            patch(
                "src.trading.rule_execution.evaluate_correlation_gate",
                return_value=MagicMock(is_allowed=True, reason=None),
            ),
            patch("src.trading.rule_execution.get_symbol_sector", return_value="sec"),
            patch(
                "src.trading.rule_execution.get_ticker",
                side_effect=lambda market, symbol: f"{symbol}.T",
            ),
            patch("src.infrastructure.discord_notification_adapter.DiscordNotificationAdapter"),
            patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter"),
            patch("src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink"),
            patch("src.reporting.discord.discord_utils.send_rule_daily_signals"),
            self.assertLogs("src.orchestration.jobs.daily", level="ERROR"),
            self.assertRaises(RuntimeError),
        ):
            daily.run_daily_rule_signals()
        broker.send_order.assert_called_once_with(
            symbol="1000.T", side=OrderSide.SELL, qty=100, order_type=OrderType.MARKET
        )

    def test_half_or_less_places_orders_and_logs_unavailable_count(self):
        with self.assertLogs("src.orchestration.jobs.daily", level="INFO") as logs:
            execute = self._run(_batch(n_ok=2, n_unavailable=2))
        execute.assert_called_once()
        self.assertEqual(len(execute.call_args.kwargs["signals"]), 2)
        self.assertTrue(execute.call_args.kwargs["allow_new_buys"])
        done = [line for line in logs.output if "日次ルールシグナル完了" in line]
        self.assertEqual(len(done), 1)
        self.assertIn("取れなかった=2", done[0])
        self.assertIn("対象=4", done[0])


class TestRunDailyRuleSignalsOrderErrors(unittest.TestCase):
    """注文の失敗が1件以上なら、知らせを送った後にジョブを失敗にする（#1075）。"""

    def _run(self, stats: dict[str, int]) -> tuple[MagicMock, MagicMock]:
        from src.orchestration.jobs import daily

        execute = MagicMock(return_value=stats)
        notify = MagicMock()
        with (
            patch(
                "src.rule_engine.pipeline.generate_rule_signals",
                return_value=_batch(n_ok=2, n_unavailable=0),
            ),
            patch("src.trading.rule_execution.execute_rule_paper_trades", execute),
            patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter"),
            patch("src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink"),
            patch("src.reporting.discord.discord_utils.send_rule_daily_signals", notify),
        ):
            daily.run_daily_rule_signals()
        return execute, notify

    def test_raises_after_notifying_when_order_failed(self):
        stats = {"buy_orders": 0, "sell_orders": 1, "skipped": 0, "errors": 1}
        notify = MagicMock()
        execute = MagicMock(return_value=stats)
        from src.orchestration.jobs import daily

        with (
            patch(
                "src.rule_engine.pipeline.generate_rule_signals",
                return_value=_batch(n_ok=2, n_unavailable=0),
            ),
            patch("src.trading.rule_execution.execute_rule_paper_trades", execute),
            patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter"),
            patch("src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink"),
            patch("src.reporting.discord.discord_utils.send_rule_daily_signals", notify),
            self.assertLogs("src.orchestration.jobs.daily", level="ERROR"),
            self.assertRaises(RuntimeError),
        ):
            daily.run_daily_rule_signals()
        notify.assert_called_once()
        # ジョブの中では発注を呼び直さない（呼び直しは run_scheduler の max_retries: 0 で止める）
        execute.assert_called_once()

    def test_succeeds_when_no_order_failed(self):
        execute, notify = self._run({"buy_orders": 1, "sell_orders": 1, "skipped": 0, "errors": 0})
        execute.assert_called_once()
        notify.assert_called_once()


class TestDailyRuleSignalsScheduleIsNotRetried(unittest.TestCase):
    def test_daily_rule_signals_has_no_retry(self):
        """ジョブを失敗にしても発注を出し直さないよう、呼び直しを付けない（#836・#1075）。"""
        import run_scheduler

        config = run_scheduler.SCHEDULE_CONFIG["daily_rule_signals"]
        self.assertEqual(config["max_retries"], 0)
        self.assertEqual(config["max_executions_per_period"], 1)
        self.assertFalse(config.get("rerun_if_interrupted", False))


if __name__ == "__main__":
    unittest.main()
