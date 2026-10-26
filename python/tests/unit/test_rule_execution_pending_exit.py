"""ユニットテスト: 16:30 のルールの道が約定待ちの SELL のある銘柄に2本目の SELL を出さない（#1169）

朝の約定で据え置かれた SELL が残る銘柄に売りシグナルが出ると、ルールの道は保有の全数で
2本目の SELL を出していた。約定待ちの無い保有の売りは今までどおり出す。
"""

import unittest
from unittest.mock import MagicMock, patch

from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.rule_execution import execute_rule_paper_trades
from src.utils.db._connection import _db_connection


def _broker(positions: list[dict]) -> MagicMock:
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_positions.return_value = positions
    broker.send_order.return_value = {"order_id": "ord-1169", "status": "pending"}
    return broker


def _long(symbol: str) -> dict:
    return {"symbol": symbol, "qty": 100, "avg_price": 1000.0, "current_price": 900.0}


class TestRuleSellSkipsPendingExit(unittest.TestCase):
    def setUp(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")
        self.addCleanup(self._clear)

    def _clear(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")

    def _pending(self, symbol: str, side: OrderSide) -> None:
        with _db_connection() as con:
            con.execute(
                "INSERT INTO paper_orders "
                "(order_id, market, symbol, side, qty, price, order_type, status, created_at) "
                "VALUES (%s, 'jp', %s, %s, 100, 0.0, %s, 'pending', CURRENT_TIMESTAMP)",
                [f"p-{symbol}-{int(side)}", symbol, int(side), int(OrderType.MARKET)],
            )

    def _run(self, broker: MagicMock, signals: list[dict]) -> dict[str, int]:
        risk = MagicMock()
        risk.evaluate_trading_gate.return_value = MagicMock(is_allowed=True, reason=None)
        risk.calc_position_size.return_value = 100
        with (
            patch("src.trading.rule_execution.RiskManager", return_value=risk),
            patch(
                "src.trading.rule_execution.get_ticker",
                side_effect=lambda market, symbol: f"{symbol}.T",
            ),
        ):
            return execute_rule_paper_trades(
                signals, "jp", trade_diff_sink=InMemoryTradeDiffSink(), broker=broker
            )

    def test_pending_sell_gets_no_second_rule_sell(self):
        self._pending("9000", OrderSide.SELL)
        broker = _broker([_long("9000"), _long("7203")])
        signals = [
            {"symbol": "9000", "signal": -1, "price": 900.0},
            {"symbol": "7203", "signal": -1, "price": 900.0},
        ]

        with self.assertLogs("src.trading.execution.pending", level="INFO") as logs:
            result = self._run(broker, signals)

        sold = [c.kwargs["symbol"] for c in broker.send_order.call_args_list]
        self.assertEqual(sold, ["7203.T"])
        self.assertEqual(result["sell_orders"], 1)
        self.assertEqual(result["skipped"], 1)
        self.assertTrue(any("9000" in line for line in logs.output))

    def test_pending_buy_does_not_block_rule_sell(self):
        self._pending("9000", OrderSide.BUY)
        broker = _broker([_long("9000")])

        result = self._run(broker, [{"symbol": "9000", "signal": -1, "price": 900.0}])

        broker.send_order.assert_called_once_with(
            symbol="9000.T", side=OrderSide.SELL, qty=100, order_type=OrderType.MARKET
        )
        self.assertEqual(result["sell_orders"], 1)


if __name__ == "__main__":
    unittest.main()
