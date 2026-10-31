"""ユニットテスト: 空売りの保有が読めない日は例外を上へ出す（#1458）

get_short_positions の中で落ちた AttributeError を0件にのみこむと、保有の上限・
空売り中の銘柄の除外・総資産から空売りが黙って抜けて新規を出す。
例外を上へ出し、16:30 の道では新規の BUY を止めて保有の SELL だけ出してから失敗にする。
"""

import logging
import unittest
from unittest.mock import MagicMock, patch

from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.position_valuation import (
    broker_positions_value,
    count_short_positions,
    short_held_codes,
)
from src.trading.rule_execution import execute_rule_paper_trades


def _broken_broker() -> MagicMock:
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "mock"
    broker.get_positions.return_value = [
        {"symbol": "1000", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0}
    ]
    broker.get_short_positions.side_effect = AttributeError("short read down")
    broker.send_order.return_value = {"order_id": "ord-1458", "status": "pending"}
    return broker


class TestPositionValuationRaises(unittest.TestCase):
    def test_count_short_positions_raises(self):
        with self.assertRaisesRegex(AttributeError, "short read down"):
            count_short_positions(_broken_broker())

    def test_short_held_codes_raises(self):
        with self.assertRaisesRegex(AttributeError, "short read down"):
            short_held_codes(_broken_broker())

    def test_broker_positions_value_raises(self):
        with self.assertRaisesRegex(AttributeError, "short read down"):
            broker_positions_value(_broken_broker(), logging.getLogger(__name__))


class TestRuleExecutionShortReadError(unittest.TestCase):
    def test_sells_then_raises_without_buys(self):
        risk = MagicMock()
        risk.evaluate_trading_gate.return_value = MagicMock(is_allowed=True, reason=None)
        risk.calc_position_size.return_value = 100
        broker = _broken_broker()
        signals = [
            {"symbol": "1000", "signal": -1, "price": 1000.0},
            {"symbol": "3000", "signal": 1, "price": 1000.0},
        ]
        with (
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
            self.assertRaisesRegex(AttributeError, "short read down"),
        ):
            execute_rule_paper_trades(
                signals, "jp", trade_diff_sink=InMemoryTradeDiffSink(), broker=broker
            )
        broker.send_order.assert_called_once_with(
            symbol="1000.T", side=OrderSide.SELL, qty=100, order_type=OrderType.MARKET
        )


if __name__ == "__main__":
    unittest.main()
