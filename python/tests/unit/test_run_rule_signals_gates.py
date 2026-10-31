"""手で流すルールの発注の CLI が日次の道と同じ門を通ることの検証（#1446）。"""

import unittest
from unittest.mock import MagicMock, patch

from src.rule_engine.types import RuleSignalBatch, UnavailableRuleSymbol

_SELL_SIGNAL = {
    "symbol": "7203",
    "rule": "dummy_rule",
    "price": 100.0,
    "win_rate": 0.5,
    "net_profit": 0.0,
    "signal": -1,
}


def _unavailable(n: int) -> list[UnavailableRuleSymbol]:
    return [
        UnavailableRuleSymbol(symbol=f"S{i}", rule="dummy_rule", reason="no_data") for i in range(n)
    ]


class TestRunRuleSignalsGates(unittest.TestCase):
    def _run(self, batch: RuleSignalBatch, trade_stats: dict, argv_extra: list[str] | None = None):
        import run_rule_signals

        execute = MagicMock(return_value=trade_stats)
        argv = ["run_rule_signals.py", "--market", "jp", "--no-discord"] + (argv_extra or [])
        with patch("run_rule_signals.YFinanceMarketDataAdapter") as adapter_cls, patch(
            "run_rule_signals.generate_rule_signals", return_value=batch
        ), patch("run_rule_signals.execute_rule_paper_trades", execute), patch(
            "run_rule_signals.make_order_broker"
        ), patch(
            "run_rule_signals.PostgresTradeDiffSink"
        ), patch(
            "sys.argv", argv
        ):
            try:
                run_rule_signals.main()
                exited = None
            except SystemExit as exc:
                exited = exc
        return execute, adapter_cls.return_value, exited

    def test_mostly_unavailable_stops_new_buys_and_fails_after_sell(self):
        batch = RuleSignalBatch(signals=[_SELL_SIGNAL], unavailable=_unavailable(9), total=10)
        execute, _, exited = self._run(batch, {"buy_orders": 0, "sell_orders": 1, "errors": 0})

        execute.assert_called_once()
        self.assertIs(execute.call_args.kwargs["allow_new_buys"], False)
        self.assertIsNotNone(exited)
        self.assertNotEqual(exited.code, 0)

    def test_market_data_port_is_passed_to_execution(self):
        batch = RuleSignalBatch(signals=[_SELL_SIGNAL], unavailable=[], total=1)
        execute, adapter, exited = self._run(
            batch, {"buy_orders": 0, "sell_orders": 1, "errors": 0}
        )

        port = execute.call_args.kwargs.get("market_data_port")
        self.assertIsNotNone(port)
        self.assertIs(port, adapter)
        self.assertIs(execute.call_args.kwargs["allow_new_buys"], True)
        self.assertIsNone(exited)

    def test_order_error_exits_nonzero(self):
        batch = RuleSignalBatch(signals=[_SELL_SIGNAL], unavailable=[], total=1)
        _, _, exited = self._run(batch, {"buy_orders": 0, "sell_orders": 0, "errors": 1})

        self.assertIsNotNone(exited)
        self.assertNotEqual(exited.code, 0)

    def test_dry_run_does_not_execute(self):
        batch = RuleSignalBatch(signals=[_SELL_SIGNAL], unavailable=[], total=1)
        execute, _, exited = self._run(
            batch, {"buy_orders": 0, "sell_orders": 0, "errors": 0}, ["--dry-run"]
        )

        execute.assert_not_called()
        self.assertIsNone(exited)


if __name__ == "__main__":
    unittest.main()
