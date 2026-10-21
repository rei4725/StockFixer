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

    def test_does_not_place_orders_when_more_than_half_unavailable(self):
        execute = MagicMock()
        from src.orchestration.jobs import daily

        with (
            patch(
                "src.rule_engine.pipeline.generate_rule_signals",
                return_value=_batch(n_ok=0, n_unavailable=3),
            ),
            patch("src.trading.rule_execution.execute_rule_paper_trades", execute),
            patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter"),
            patch("src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink"),
            self.assertRaises(RuntimeError),
        ):
            daily.run_daily_rule_signals()
        execute.assert_not_called()

    def test_half_or_less_places_orders_and_logs_unavailable_count(self):
        with self.assertLogs("src.orchestration.jobs.daily", level="INFO") as logs:
            execute = self._run(_batch(n_ok=2, n_unavailable=2))
        execute.assert_called_once()
        self.assertEqual(len(execute.call_args.kwargs["signals"]), 2)
        done = [line for line in logs.output if "日次ルールシグナル完了" in line]
        self.assertEqual(len(done), 1)
        self.assertIn("取れなかった=2", done[0])
        self.assertIn("対象=4", done[0])


if __name__ == "__main__":
    unittest.main()
