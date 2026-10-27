"""ユニットテスト: 16:30 のルールの道で新規の BUY の手順が落ちても保有の SELL は出す（#1221）

update_peak_balance・evaluate_trading_gate・相関ゲートは新規の BUY にだけ使う。
これらが例外を出した日も売りシグナルの出た保有は売り、その後に同じ例外を投げ直す。
"""

import unittest
from unittest.mock import MagicMock, patch

from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.rule_execution import execute_rule_paper_trades


def _long(symbol: str) -> dict:
    return {"symbol": symbol, "qty": 100, "avg_price": 1000.0, "current_price": 1000.0}


_SIGNALS = [
    {"symbol": "1000", "signal": -1, "price": 1000.0},
    {"symbol": "3000", "signal": 1, "price": 1000.0},
]


class TestRuleExecutionBuyStepErrorKeepsSells(unittest.TestCase):
    def _run(self, broker: MagicMock, risk: MagicMock, corr: MagicMock) -> None:
        with (
            patch("src.trading.rule_execution.RiskManager", return_value=risk),
            patch("src.trading.rule_execution.evaluate_correlation_gate", corr),
            patch("src.trading.rule_execution.get_symbol_sector", return_value="sec"),
            patch(
                "src.trading.rule_execution.get_ticker",
                side_effect=lambda market, symbol: f"{symbol}.T",
            ),
        ):
            execute_rule_paper_trades(
                _SIGNALS, "jp", trade_diff_sink=InMemoryTradeDiffSink(), broker=broker
            )

    def _broker(self) -> MagicMock:
        broker = MagicMock(spec=BrokerBase)
        broker.broker_name = "mock"
        broker.get_positions.return_value = [_long("1000")]
        broker.send_order.return_value = {"order_id": "ord-1221", "status": "pending"}
        return broker

    def _risk(self) -> MagicMock:
        risk = MagicMock()
        risk.evaluate_trading_gate.return_value = MagicMock(is_allowed=True, reason=None)
        risk.calc_position_size.return_value = 100
        return risk

    def _corr(self) -> MagicMock:
        return MagicMock(return_value=MagicMock(is_allowed=True, reason=None))

    def _assert_sell_only_and_raises(
        self, broker: MagicMock, risk: MagicMock, corr: MagicMock, message: str
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, message):
            self._run(broker, risk, corr)
        broker.send_order.assert_called_once_with(
            symbol="1000.T", side=OrderSide.SELL, qty=100, order_type=OrderType.MARKET
        )

    def test_trading_gate_error(self):
        risk = self._risk()
        risk.evaluate_trading_gate.side_effect = RuntimeError("gate down")
        self._assert_sell_only_and_raises(self._broker(), risk, self._corr(), "gate down")

    def test_update_peak_balance_error(self):
        risk = self._risk()
        risk.update_peak_balance.side_effect = RuntimeError("peak down")
        self._assert_sell_only_and_raises(self._broker(), risk, self._corr(), "peak down")

    def test_correlation_gate_error(self):
        corr = MagicMock(side_effect=RuntimeError("corr down"))
        self._assert_sell_only_and_raises(self._broker(), self._risk(), corr, "corr down")

    def test_positions_read_error_places_no_orders(self):
        broker = self._broker()
        broker.get_positions.side_effect = RuntimeError("positions down")
        with self.assertRaisesRegex(RuntimeError, "positions down"):
            self._run(broker, self._risk(), self._corr())
        broker.send_order.assert_not_called()


if __name__ == "__main__":
    unittest.main()
