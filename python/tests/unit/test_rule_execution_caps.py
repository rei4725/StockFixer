"""ユニットテスト: 16:30 のルールの道の発注の上限と分かれ道（#1447）

1回の発注上限（MAX_ORDERS_PER_RUN）・1銘柄の予算の上限（initial_budget_per_trade）・
空売りの保有が読めない日・約定済みの保有への BUY を、のちの直しで黙って外さないために見る。
"""

import unittest
from unittest.mock import MagicMock, patch

from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.rule_execution import execute_rule_paper_trades


def _long(symbol: str) -> dict:
    return {"symbol": symbol, "qty": 100, "avg_price": 1000.0, "current_price": 1000.0}


def _broker(positions: list[dict]) -> MagicMock:
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "mock"
    broker.get_positions.return_value = positions
    broker.send_order.return_value = {"order_id": "ord-1447", "status": "pending"}
    return broker


class TestRuleExecutionCaps(unittest.TestCase):
    def _run(
        self,
        broker: MagicMock,
        signals: list[dict],
        *,
        max_orders: int = 10,
        position_size: int = 100,
        budget: float = 100_000,
    ) -> dict[str, int]:
        risk = MagicMock()
        risk.evaluate_trading_gate.return_value = MagicMock(is_allowed=True, reason=None)
        risk.calc_position_size.return_value = position_size
        corr = MagicMock(return_value=MagicMock(is_allowed=True, reason=None))
        with (
            patch("src.trading.rule_execution.RiskManager", return_value=risk),
            patch("src.trading.rule_execution.evaluate_correlation_gate", corr),
            patch(
                "src.trading.rule_execution.get_symbol_sector",
                side_effect=lambda market, s: f"sec-{s}",
            ),
            patch("src.trading.rule_execution.MAX_POSITIONS", 10),
            patch("src.trading.rule_execution.MAX_SECTOR_POSITIONS", 10),
            patch("src.trading.rule_execution.MAX_ORDERS_PER_RUN", max_orders),
            patch(
                "src.trading.rule_execution.get_ticker",
                side_effect=lambda market, symbol: f"{symbol}.T",
            ),
        ):
            return execute_rule_paper_trades(
                signals,
                "jp",
                initial_budget_per_trade=budget,
                trade_diff_sink=InMemoryTradeDiffSink(),
                broker=broker,
            )

    def _sent(self, broker: MagicMock) -> list[tuple[str, OrderSide]]:
        return [(c.kwargs["symbol"], c.kwargs["side"]) for c in broker.send_order.call_args_list]

    def test_max_orders_per_run_stops_later_buys_but_keeps_sell(self):
        broker = _broker([_long("1000")])
        signals = [
            {"symbol": "3000", "signal": 1, "price": 1000.0},
            {"symbol": "3001", "signal": 1, "price": 1000.0},
            {"symbol": "3002", "signal": 1, "price": 1000.0},
            {"symbol": "1000", "signal": -1, "price": 1000.0},
        ]

        result = self._run(broker, signals, max_orders=2)

        self.assertEqual(
            self._sent(broker),
            [("3000.T", OrderSide.BUY), ("3001.T", OrderSide.BUY), ("1000.T", OrderSide.SELL)],
        )
        self.assertEqual(result["buy_orders"], 2)
        self.assertEqual(result["sell_orders"], 1)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["errors"], 0)

    def test_budget_per_trade_caps_qty_to_lot_multiple(self):
        broker = _broker([])
        signals = [
            {"symbol": "3000", "signal": 1, "price": 1000.0},
            {"symbol": "3001", "signal": 1, "price": 1500.0},
        ]

        result = self._run(broker, signals, position_size=500, budget=100_000)

        broker.send_order.assert_called_once_with(
            symbol="3000.T", side=OrderSide.BUY, qty=100, order_type=OrderType.MARKET
        )
        self.assertEqual(result["buy_orders"], 1)
        self.assertEqual(result["skipped"], 1)

    def test_short_positions_read_error_places_sell_only_and_raises(self):
        broker = _broker([_long("1000")])
        broker.get_short_positions.side_effect = RuntimeError("shorts down")
        signals = [
            {"symbol": "3000", "signal": 1, "price": 1000.0},
            {"symbol": "1000", "signal": -1, "price": 1000.0},
        ]

        with self.assertRaisesRegex(RuntimeError, "shorts down"):
            self._run(broker, signals)

        broker.send_order.assert_called_once_with(
            symbol="1000.T", side=OrderSide.SELL, qty=100, order_type=OrderType.MARKET
        )

    def test_buy_signal_on_held_symbol_is_skipped(self):
        broker = _broker([_long("1000")])

        result = self._run(broker, [{"symbol": "1000", "signal": 1, "price": 1000.0}])

        broker.send_order.assert_not_called()
        self.assertEqual(result["buy_orders"], 0)
        self.assertEqual(result["skipped"], 1)


if __name__ == "__main__":
    unittest.main()
