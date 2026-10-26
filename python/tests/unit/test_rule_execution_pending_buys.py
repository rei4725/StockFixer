"""ユニットテスト: 16:30 のルールの道が約定待ちの BUY を保有上限・業種の上限・相関ゲートに数える（#1176）

紙の BUY は朝の約定まで get_positions に出ないため、ルールの道は約定待ちの買いを数えずに
保有上限と業種の上限を超えて建てていた。読む関数が無い・読めないときは約定済みの保有だけで続ける。
"""

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.rule_execution import execute_rule_paper_trades


def _long(symbol: str) -> dict:
    return {"symbol": symbol, "qty": 100, "avg_price": 1000.0, "current_price": 1000.0}


def _broker(positions: list[dict], pending_buys: Any = None) -> MagicMock:
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "mock"
    broker.get_positions.return_value = positions
    broker.send_order.return_value = {"order_id": "ord-1176", "status": "pending"}
    if isinstance(pending_buys, Exception):
        broker.get_pending_buy_symbols = MagicMock(side_effect=pending_buys)
    elif pending_buys is not None:
        broker.get_pending_buy_symbols = MagicMock(return_value=pending_buys)
    return broker


class TestRuleExecutionCountsPendingBuys(unittest.TestCase):
    def _run(
        self,
        broker: MagicMock,
        signals: list[dict],
        *,
        max_positions: int = 10,
        max_sector: int = 10,
        sectors: dict[str, str] | None = None,
    ) -> tuple[dict[str, int], MagicMock]:
        risk = MagicMock()
        risk.evaluate_trading_gate.return_value = MagicMock(is_allowed=True, reason=None)
        risk.calc_position_size.return_value = 100
        corr = MagicMock(return_value=MagicMock(is_allowed=True, reason=None))
        sector_map = sectors or {}
        with (
            patch("src.trading.rule_execution.RiskManager", return_value=risk),
            patch("src.trading.rule_execution.evaluate_correlation_gate", corr),
            patch(
                "src.trading.rule_execution.get_symbol_sector",
                side_effect=lambda market, s: sector_map.get(s, f"sec-{s}"),
            ),
            patch("src.trading.rule_execution.MAX_POSITIONS", max_positions),
            patch("src.trading.rule_execution.MAX_SECTOR_POSITIONS", max_sector),
            patch("src.trading.rule_execution.MAX_ORDERS_PER_RUN", 10),
            patch(
                "src.trading.rule_execution.get_ticker",
                side_effect=lambda market, symbol: f"{symbol}.T",
            ),
        ):
            result = execute_rule_paper_trades(
                signals, "jp", trade_diff_sink=InMemoryTradeDiffSink(), broker=broker
            )
        return result, corr

    def test_nine_held_plus_one_pending_buy_blocks_new_buy(self):
        held = [_long(str(1000 + i)) for i in range(9)]
        broker = _broker(held, pending_buys={"2000"})

        result, _ = self._run(broker, [{"symbol": "3000", "signal": 1, "price": 1000.0}])

        broker.send_order.assert_not_called()
        self.assertEqual(result["buy_orders"], 0)
        self.assertEqual(result["skipped"], 1)

    def test_pending_buy_symbol_is_not_bought_again(self):
        broker = _broker([], pending_buys={"2000"})

        result, _ = self._run(broker, [{"symbol": "2000", "signal": 1, "price": 1000.0}])

        broker.send_order.assert_not_called()
        self.assertEqual(result["buy_orders"], 0)

    def test_pending_buy_counts_toward_sector_cap_and_correlation_gate(self):
        broker = _broker([_long("1000")], pending_buys={"2000.T"})
        sectors = {"1000": "A", "2000": "B", "3000": "B", "4000": "C"}
        signals = [
            {"symbol": "3000", "signal": 1, "price": 1000.0},
            {"symbol": "4000", "signal": 1, "price": 1000.0},
        ]

        result, corr = self._run(broker, signals, max_sector=1, sectors=sectors)

        bought = [c.kwargs["symbol"] for c in broker.send_order.call_args_list]
        self.assertEqual(bought, ["4000.T"])
        self.assertEqual(corr.call_args.args[0], ["1000", "2000"])

    def test_broker_without_reader_keeps_held_only(self):
        held = [_long(str(1000 + i)) for i in range(9)]
        broker = _broker(held)

        result, _ = self._run(broker, [{"symbol": "3000", "signal": 1, "price": 1000.0}])

        broker.send_order.assert_called_once_with(
            symbol="3000.T", side=OrderSide.BUY, qty=100, order_type=OrderType.MARKET
        )
        self.assertEqual(result["buy_orders"], 1)

    def test_reader_failure_keeps_held_only(self):
        held = [_long(str(1000 + i)) for i in range(9)]
        broker = _broker(held, pending_buys=RuntimeError("db down"))

        result, _ = self._run(broker, [{"symbol": "3000", "signal": 1, "price": 1000.0}])

        self.assertEqual(result["buy_orders"], 1)
        self.assertEqual(result["errors"], 0)

    def test_sell_branch_uses_settled_positions_only(self):
        broker = _broker([_long("1000")], pending_buys={"2000"})
        signals = [
            {"symbol": "1000", "signal": -1, "price": 1000.0},
            {"symbol": "2000", "signal": -1, "price": 1000.0},
        ]

        result, _ = self._run(broker, signals)

        broker.send_order.assert_called_once_with(
            symbol="1000.T", side=OrderSide.SELL, qty=100, order_type=OrderType.MARKET
        )
        self.assertEqual(result["sell_orders"], 1)
        self.assertEqual(result["skipped"], 1)


class TestRuleExecutionAllowNewBuys(unittest.TestCase):
    """allow_new_buys=False では新規の BUY だけ止め、保有の SELL は出す（#1191）。"""

    def test_blocks_buys_but_places_sells(self):
        broker = _broker([_long("1000")])
        risk = MagicMock()
        risk.evaluate_trading_gate.return_value = MagicMock(is_allowed=True, reason=None)
        risk.calc_position_size.return_value = 100
        corr = MagicMock(return_value=MagicMock(is_allowed=True, reason=None))
        signals = [
            {"symbol": "1000", "signal": -1, "price": 1000.0},
            {"symbol": "3000", "signal": 1, "price": 1000.0},
        ]
        with (
            patch("src.trading.rule_execution.RiskManager", return_value=risk),
            patch("src.trading.rule_execution.evaluate_correlation_gate", corr),
            patch("src.trading.rule_execution.get_symbol_sector", return_value="sec"),
            patch(
                "src.trading.rule_execution.get_ticker",
                side_effect=lambda market, symbol: f"{symbol}.T",
            ),
        ):
            result = execute_rule_paper_trades(
                signals,
                "jp",
                trade_diff_sink=InMemoryTradeDiffSink(),
                broker=broker,
                allow_new_buys=False,
            )

        broker.send_order.assert_called_once_with(
            symbol="1000.T", side=OrderSide.SELL, qty=100, order_type=OrderType.MARKET
        )
        corr.assert_not_called()
        self.assertEqual(result["buy_orders"], 0)
        self.assertEqual(result["sell_orders"], 1)
        self.assertEqual(result["skipped"], 1)


if __name__ == "__main__":
    unittest.main()
