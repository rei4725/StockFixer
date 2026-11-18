"""ユニットテスト: 16:30 のルールの道の記録に、株数0で飛ばした新規買いの銘柄の一覧を写す（#1997）

記録の JSON の zero_qty_symbols に、銘柄ごとに reason（skip_reasons の鍵）・price・
lot_cost・sized_qty・budget を1行ずつ入れ、zero_qty_count はその数で、
skip_reasons の株数0の鍵の和と一致する。株数0が無い日は空のリストと 0。
"""

import json
import os
import unittest
from datetime import datetime
from typing import Optional
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.trading.brokers.base import BrokerBase, OrderSide
from src.trading.rule_execution import execute_rule_paper_trades
from src.utils.data_path_utils import get_results_dir

_NOW = datetime(2026, 11, 17, 16, 30, tzinfo=ZoneInfo("Asia/Tokyo"))

# 5803 は予算の上限で0、6000 は単元未満で0、7000 は買える
_SIGNALS = [
    {"symbol": "5803", "signal": 1, "price": 1500.0},
    {"symbol": "6000", "signal": 1, "price": 500.0},
    {"symbol": "7000", "signal": 1, "price": 200.0},
]
_SIZES = {"5803": 500, "6000": 0, "7000": 100}
_UNDER_LOT = "単元未満 投資額=30000 1単元の代金=50000 lot=100"


def _broker() -> MagicMock:
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "mock"
    broker.get_positions.return_value = []
    broker.get_short_positions = MagicMock(return_value=[])
    broker.get_pending_buy_symbols = MagicMock(return_value=set())
    broker.send_order.return_value = {"order_id": "ord-1997", "status": "pending"}
    return broker


def _read() -> dict:
    path = os.path.join(get_results_dir(), "orders", "rule", "2026-11-17.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class TestRuleZeroQtySymbolsRecord(unittest.TestCase):
    def _run(self, broker: MagicMock, signals: list, zero_reason: Optional[str]):
        risk = MagicMock()
        risk.evaluate_trading_gate.return_value = MagicMock(is_allowed=True, reason=None)
        risk.calc_position_size.side_effect = lambda symbol, price: _SIZES[symbol]
        risk.zero_reason = zero_reason
        corr = MagicMock(return_value=MagicMock(is_allowed=True, reason=None, enc=None))
        with (
            patch("src.trading.rule_execution.RiskManager", return_value=risk),
            patch("src.trading.rule_execution.evaluate_correlation_gate", corr),
            patch("src.trading.rule_execution.get_symbol_sector", return_value="A"),
            patch("src.trading.rule_execution.MAX_POSITIONS", 10),
            patch("src.trading.rule_execution.MAX_SECTOR_POSITIONS", 10),
            patch("src.trading.rule_execution.MAX_ORDERS_PER_RUN", 10),
            patch(
                "src.trading.rule_execution.get_ticker",
                side_effect=lambda market, symbol: f"{symbol}.T",
            ),
            patch("src.trading.execution.gate_record.now_jst", return_value=_NOW),
        ):
            return execute_rule_paper_trades(
                signals,
                "jp",
                trade_diff_sink=InMemoryTradeDiffSink(),
                broker=broker,
                initial_budget_per_trade=100_000,
                record_gate=True,
            )

    def test_zero_qty_buys_are_listed_per_symbol_with_reason_and_price(self):
        broker = _broker()

        result = self._run(broker, _SIGNALS, zero_reason=_UNDER_LOT)
        record = _read()

        self.assertEqual(
            record["zero_qty_symbols"],
            [
                {
                    "symbol": "5803",
                    "reason": "zero_qty_budget_cap",
                    "price": 1500.0,
                    "lot_cost": 150000.0,
                    "sized_qty": 500,
                    "budget": 100000.0,
                },
                {
                    "symbol": "6000",
                    "reason": "zero_qty_under_lot",
                    "price": 500.0,
                    "lot_cost": 50000.0,
                    "sized_qty": 0,
                    "budget": 100000.0,
                },
            ],
        )
        zero_sum = sum(n for k, n in record["skip_reasons"].items() if k.startswith("zero_qty"))
        self.assertEqual(record["zero_qty_count"], 2)
        self.assertEqual(record["zero_qty_count"], zero_sum)
        # 既存の欄と発注は変わらない
        self.assertEqual(
            record["skip_reasons"], {"zero_qty_budget_cap": 1, "zero_qty_under_lot": 1}
        )
        self.assertEqual(result["buy_orders"], 1)
        self.assertEqual(result["skipped"], 2)
        broker.send_order.assert_called_once()
        kwargs = broker.send_order.call_args.kwargs
        self.assertEqual(
            (kwargs["symbol"], kwargs["side"], kwargs["qty"]), ("7000.T", OrderSide.BUY, 100)
        )

    def test_day_without_zero_qty_has_empty_list_and_zero(self):
        broker = _broker()

        self._run(broker, [_SIGNALS[2]], zero_reason=None)
        record = _read()

        self.assertEqual(record["zero_qty_symbols"], [])
        self.assertEqual(record["zero_qty_count"], 0)


if __name__ == "__main__":
    unittest.main()
