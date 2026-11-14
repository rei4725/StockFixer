"""
ユニットテスト: 8:50 の新規で、1回の発注の上限と業種の枠を発注した銘柄だけに数える（#1759）

株数0などで飛ばした候補が上限と業種の枠を使い切り、後ろの買える候補が見られないまま
終わらないことを確かめる（16:30 の #1610 の直しと同じ形）。
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from tests.unit.test_order_skip_reasons import (
    _ALLOWED,
    _CORR_OK,
    _HELD,
    _NOW,
    _PREDICTIONS,
    _attach,
    _record_path,
    _row,
    _score,
)

from src.infrastructure.in_memory import InMemoryOrderRunSink, InMemoryTradeDiffSink
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.execution import run_daily_orders, selection
from src.trading.risk_manager import RiskManager

_MAX_ORDERS = 5


def _sector(market, symbol):
    """3 で始まる銘柄は同じ業種、ほかは銘柄ごとに別の業種にする"""
    return "SAME" if symbol.startswith("3") else f"UNKNOWN:{market}:{symbol}"


class TestOrderBuyCapCountsSent(unittest.TestCase):
    def _run(self, frame, zero_qty, *, short_side=False, max_sector=3):
        if os.path.exists(_record_path()):
            os.remove(_record_path())
        sl_risk = MagicMock()
        sl_risk.return_value.check_sl_tp.return_value = (False, False, "")
        broker = MagicMock(spec=BrokerBase)
        broker.broker_name = "paper"
        broker.get_balance.return_value = 1_000_000.0
        broker.get_positions.return_value = [dict(_HELD)]
        broker.get_short_positions = MagicMock(return_value=[])
        broker.send_order.return_value = {"order_id": "test-001", "status": "pending"}

        def _qty(self_, symbol, *args, **kwargs):
            return 0 if symbol in zero_qty else 100

        patches = [
            patch.object(_PREDICTIONS, "get_latest_by_market", return_value=frame),
            patch("src.trading.execution.runner._record_order"),
            patch("src.trading.execution.sl_tp._record_order"),
            patch(
                "src.trading.execution.runner._choose_order_params",
                return_value=(OrderType.MARKET, 0.0, "market", "open"),
            ),
            patch("src.trading.execution.sl_tp.RiskManager", sl_risk),
            patch("src.trading.execution.RiskManager.evaluate_trading_gate", return_value=_ALLOWED),
            patch.object(RiskManager, "calc_position_size", autospec=True, side_effect=_qty),
            patch("src.trading.execution.RiskManager.update_peak_balance"),
            patch("src.trading.execution.runner.evaluate_correlation_gate", return_value=_CORR_OK),
            patch("src.trading.execution.runner._load_exit_model", return_value=None),
            patch("src.trading.execution.params.get_optimal_params", return_value={}),
            patch("src.trading.execution.runner._attach_dynamic_thresholds", side_effect=_attach),
            patch(
                "src.trading.execution.runner.apply_multi_horizon_score_column",
                side_effect=_score,
            ),
            patch("src.trading.execution.selection.get_symbol_sector", side_effect=_sector),
            patch("src.trading.execution.runner.MAX_POSITIONS", 20),
            patch("src.trading.execution.selection.MAX_ORDERS_PER_RUN", _MAX_ORDERS),
            patch("src.trading.execution.selection.MAX_SECTOR_POSITIONS", max_sector),
            patch.object(selection._apply_buy_sector_limit, "__defaults__", (max_sector, (), "jp")),
            patch("src.trading.execution.runner.ENABLE_SHORT_SIDE", short_side),
            patch("src.trading.execution.gate_record.now_jst", return_value=_NOW),
        ]
        for p in patches:
            p.start()
        try:
            run_daily_orders(
                broker,
                order_run_sink=InMemoryOrderRunSink(),
                trade_diff_sink=InMemoryTradeDiffSink(),
                prediction_repo=_PREDICTIONS,
                market="jp",
                mode="paper",
            )
        finally:
            for p in reversed(patches):
                p.stop()
        with open(_record_path(), encoding="utf-8") as f:
            record = json.load(f)
        sent = [(c.args[0], c.args[1], c.args[2]) for c in broker.send_order.call_args_list]
        return record, sent

    def test_zero_qty_head_does_not_use_up_max_orders(self):
        """先頭の5件が株数0でも、6件目と7件目の BUY が出る"""
        frame = pd.DataFrame([_row(f"100{i}", 0.30 - 0.01 * i) for i in range(7)])
        zero = {f"100{i}" for i in range(5)}
        record, sent = self._run(frame, zero)
        self.assertEqual(sent, [("1005", OrderSide.BUY, 100), ("1006", OrderSide.BUY, 100)])
        self.assertEqual(sum(record["skip_reasons"].values()), record["skipped"])

    def test_zero_qty_does_not_use_up_sector_cap(self):
        """同じ業種で株数0の2件の後の株数の出る1件は、業種の上限2でも BUY になる"""
        frame = pd.DataFrame([_row("3001", 0.30), _row("3002", 0.29), _row("3003", 0.28)])
        record, sent = self._run(frame, {"3001", "3002"}, max_sector=2)
        self.assertEqual(sent, [("3003", OrderSide.BUY, 100)])

    def test_sector_cap_counts_sent_buys(self):
        """発注した同じ業種の2件の後の3件目は、業種の上限2で sector_cap に数える"""
        frame = pd.DataFrame([_row("3001", 0.30), _row("3002", 0.29), _row("3003", 0.28)])
        record, sent = self._run(frame, set(), max_sector=2)
        self.assertEqual(sent, [("3001", OrderSide.BUY, 100), ("3002", OrderSide.BUY, 100)])
        self.assertEqual(record["skip_reasons"], {"sector_cap": 1})

    def test_short_zero_qty_head_does_not_use_up_max_orders(self):
        """新規ショートで先頭が株数0でも後ろが建ち、建つ数は上限の5件までである"""
        frame = pd.DataFrame(
            [_row(f"200{i}", -0.20 + 0.01 * i, diff=-0.05) for i in range(_MAX_ORDERS + 2)]
        )
        record, sent = self._run(frame, {"2000"}, short_side=True)
        self.assertEqual(
            sent, [(f"200{i}", OrderSide.SHORT, 100) for i in range(1, _MAX_ORDERS + 1)]
        )
        self.assertEqual(record["skip_reasons"].get("max_orders_per_run"), 1)


if __name__ == "__main__":
    unittest.main()
