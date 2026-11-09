"""ユニットテスト: 16:30 のルールの道で、業種の枠は発注した銘柄だけに数える（#1610）

株数を計算する前に業種の上限でふるうと、株数0で飛ばす候補が枠を使い、
後ろの株数の出る候補が業種の上限で落ちていた。
"""

import unittest
from unittest.mock import MagicMock, patch

from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.rule_execution import execute_rule_paper_trades

# 1銘柄の上限 100_000 円・単元 100 株では、1000 円を超える銘柄は株数0になる
_CHEAP = 500.0
_EXPENSIVE = 2000.0


def _long(symbol: str) -> dict:
    return {"symbol": symbol, "qty": 100, "avg_price": 1000.0, "current_price": 1000.0}


def _broker(positions: list[dict], pending_buys: object = None) -> MagicMock:
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "mock"
    broker.get_positions.return_value = positions
    broker.get_short_positions = MagicMock(return_value=[])
    broker.send_order.return_value = {"order_id": "ord-1610", "status": "pending"}
    if isinstance(pending_buys, Exception):
        broker.get_pending_buy_symbols = MagicMock(side_effect=pending_buys)
    elif pending_buys is not None:
        broker.get_pending_buy_symbols = MagicMock(return_value=pending_buys)
    return broker


class TestRuleSectorCapCountsOnlyPlacedBuys(unittest.TestCase):
    def _run(self, broker: MagicMock, signals: list[dict], *, sector=None, record_gate=False):
        risk = MagicMock()
        risk.evaluate_trading_gate.return_value = MagicMock(is_allowed=True, reason=None)
        risk.calc_position_size.return_value = 1000
        corr = MagicMock(return_value=MagicMock(is_allowed=True, reason=None))
        saved: list = []
        with (
            patch("src.trading.rule_execution.RiskManager", return_value=risk),
            patch("src.trading.rule_execution.evaluate_correlation_gate", corr),
            patch(
                "src.trading.rule_execution.get_symbol_sector",
                side_effect=sector or (lambda market, s: "A"),
            ),
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
        ):
            result = execute_rule_paper_trades(
                signals,
                "jp",
                trade_diff_sink=InMemoryTradeDiffSink(),
                broker=broker,
                record_gate=record_gate,
            )
        return result, saved

    def test_zero_qty_candidates_do_not_use_sector_slots(self):
        broker = _broker([])
        signals = [
            {"symbol": "1001", "signal": 1, "price": _EXPENSIVE},
            {"symbol": "1002", "signal": 1, "price": _EXPENSIVE},
            {"symbol": "1003", "signal": 1, "price": _CHEAP},
        ]

        result, saved = self._run(broker, signals, record_gate=True)

        broker.send_order.assert_called_once_with(
            symbol="1003.T", side=OrderSide.BUY, qty=200, order_type=OrderType.MARKET
        )
        self.assertEqual(result["buy_orders"], 1)
        self.assertEqual(saved[0].skip_reasons, {"zero_qty": 2})

    def test_three_buyable_candidates_only_first_two_and_third_is_sector_cap(self):
        broker = _broker([])
        signals = [{"symbol": s, "signal": 1, "price": _CHEAP} for s in ("1001", "1002", "1003")]

        result, saved = self._run(broker, signals, record_gate=True)

        bought = [c.kwargs["symbol"] for c in broker.send_order.call_args_list]
        self.assertEqual(bought, ["1001.T", "1002.T"])
        self.assertEqual(result["buy_orders"], 2)
        self.assertEqual(saved[0].skip_reasons, {"sector_cap": 1})

    def test_sector_filled_by_held_and_pending_blocks_buy(self):
        broker = _broker([_long("2000")], pending_buys={"2001.T"})
        signals = [
            {"symbol": "1001", "signal": 1, "price": _CHEAP},
            {"symbol": "3001", "signal": 1, "price": _CHEAP},
        ]
        sectors = {"2000": "A", "2001": "A", "1001": "A", "3001": "B"}

        result, saved = self._run(
            broker, signals, sector=lambda market, s: sectors[s], record_gate=True
        )

        bought = [c.kwargs["symbol"] for c in broker.send_order.call_args_list]
        self.assertEqual(bought, ["3001.T"])
        self.assertEqual(saved[0].skip_reasons, {"sector_cap": 1})

    def test_sector_lookup_error_stops_buys_places_sell_and_reraises(self):
        error = RuntimeError("sector lookup down")
        broker = _broker([_long("2000")])
        signals = [
            {"symbol": "1001", "signal": 1, "price": _CHEAP},
            {"symbol": "2000", "signal": -1, "price": _CHEAP},
        ]

        def _fail(market, s):
            raise error

        with self.assertRaises(RuntimeError) as caught:
            self._run(broker, signals, sector=_fail)

        self.assertIs(caught.exception, error)
        broker.send_order.assert_called_once_with(
            symbol="2000.T", side=OrderSide.SELL, qty=100, order_type=OrderType.MARKET
        )


if __name__ == "__main__":
    unittest.main()
