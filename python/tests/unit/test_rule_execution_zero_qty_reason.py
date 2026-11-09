"""ユニットテスト: 16:30 のルールの道で、株数0の理由を分けて数える（#1748）

1銘柄の予算の上限で0になったもの（calc_position_size は0でない）と、
calc_position_size が0を返したもの（risk.zero_reason の理由ごと）を、
記録の skip_reasons の別の鍵に数え、INFO の行に本当の理由を出す。
"""

import unittest
from typing import Optional
from unittest.mock import MagicMock, patch

from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.trading.brokers.base import BrokerBase
from src.trading.rule_execution import execute_rule_paper_trades


def _broker() -> MagicMock:
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "mock"
    broker.get_positions.return_value = []
    broker.get_short_positions = MagicMock(return_value=[])
    broker.get_pending_buy_symbols = MagicMock(return_value=set())
    broker.send_order.return_value = {"order_id": "ord-1748", "status": "pending"}
    return broker


class TestRuleZeroQtyReason(unittest.TestCase):
    def _run(self, broker: MagicMock, price: float, size: int, zero_reason: Optional[str]):
        risk = MagicMock()
        risk.evaluate_trading_gate.return_value = MagicMock(is_allowed=True, reason=None)
        risk.calc_position_size.return_value = size
        risk.zero_reason = zero_reason
        corr = MagicMock(return_value=MagicMock(is_allowed=True, reason=None))
        saved: list = []
        with (
            patch("src.trading.rule_execution.RiskManager", return_value=risk),
            patch("src.trading.rule_execution.evaluate_correlation_gate", corr),
            patch("src.trading.rule_execution.get_symbol_sector", return_value="A"),
            patch("src.trading.rule_execution.MAX_POSITIONS", 10),
            patch("src.trading.rule_execution.MAX_SECTOR_POSITIONS", 2),
            patch("src.trading.rule_execution.MAX_ORDERS_PER_RUN", 10),
            patch(
                "src.trading.rule_execution.get_ticker",
                side_effect=lambda market, symbol: f"{symbol}.T",
            ),
            patch(
                "src.trading.rule_execution.save_rule_gate_record",
                side_effect=lambda record, **_: saved.append(record),
            ),
            self.assertLogs("src.trading.rule_execution", level="INFO") as logs,
        ):
            result = execute_rule_paper_trades(
                [{"symbol": "5803", "signal": 1, "price": price}],
                "jp",
                trade_diff_sink=InMemoryTradeDiffSink(),
                broker=broker,
                initial_budget_per_trade=100_000,
                record_gate=True,
            )
        return result, saved, "\n".join(logs.output)

    def test_budget_cap_zero_is_counted_under_budget_cap_key(self):
        broker = _broker()

        result, saved, log = self._run(broker, price=1500.0, size=500, zero_reason=None)

        broker.send_order.assert_not_called()
        self.assertEqual(result["buy_orders"], 0)
        self.assertEqual(saved[0].skip_reasons, {"zero_qty_budget_cap": 1})
        self.assertIn("1銘柄の予算の上限", log)

    def test_kelly_zero_is_counted_under_kelly_key_with_reason_in_log(self):
        broker = _broker()

        result, saved, log = self._run(
            broker, price=1500.0, size=0, zero_reason="kelly<=0 kelly=-0.120"
        )

        broker.send_order.assert_not_called()
        self.assertEqual(saved[0].skip_reasons, {"zero_qty_kelly": 1})
        self.assertIn("kelly<=0 kelly=-0.120", log)

    def test_under_lot_zero_is_counted_under_its_own_key(self):
        broker = _broker()

        _, saved, log = self._run(
            broker,
            price=500.0,
            size=0,
            zero_reason="単元未満 投資額=30000 1単元の代金=50000 lot=100",
        )

        self.assertEqual(saved[0].skip_reasons, {"zero_qty_under_lot": 1})
        self.assertIn("単元未満", log)

    def test_unknown_reason_falls_back_to_zero_qty(self):
        broker = _broker()

        _, saved, _ = self._run(broker, price=500.0, size=0, zero_reason=None)

        self.assertEqual(saved[0].skip_reasons, {"zero_qty": 1})

    def test_buy_with_positive_qty_still_sends_order(self):
        broker = _broker()

        result, saved, _ = self._run(broker, price=500.0, size=100, zero_reason=None)

        broker.send_order.assert_called_once()
        self.assertEqual(result["buy_orders"], 1)
        self.assertEqual(saved[0].skip_reasons, {})


if __name__ == "__main__":
    unittest.main()
