"""
ユニットテスト: 8:50 の新規で業種の上限と1回の発注の上限（head）で落とした候補を数える（#1770）

業種の上限で落とした候補は sector_cap、head で落とした候補は max_orders_per_run として
skip_reasons に入れる（鍵は 16:30 の RuleGateRecord とそろえる）。skipped と skip_reasons の和は
一致し、発注される銘柄と株数は変わらない。
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
from src.trading.execution import run_daily_orders
from src.trading.risk_manager import RiskManager

_MAX_ORDERS = 5


def _sector(market, symbol):
    """3 で始まる銘柄は同じ業種、ほかは銘柄ごとに別の業種にする"""
    return "SAME" if symbol.startswith("3") else f"UNKNOWN:{market}:{symbol}"


class TestOrderSkipReasonsCaps(unittest.TestCase):
    def _run(self, frame, *, short_side=False):
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
            patch.object(RiskManager, "calc_position_size", return_value=100),
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
            patch("src.trading.execution.runner.ENABLE_SHORT_SIDE", short_side),
            patch("src.trading.execution.gate_record.now_jst", return_value=_NOW),
        ]
        for p in patches:
            p.start()
        try:
            stats = run_daily_orders(
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
        return stats, record, sent

    def test_buy_sector_cap_and_head_drops_are_counted(self):
        """業種の上限で2件・head で2件落ちた朝は sector_cap 2・max_orders_per_run 2 が出る"""
        frame = pd.DataFrame(
            [
                _row("9000", 0.0),  # 保有。売りも買いもしない
                _row("3001", 0.30),
                _row("3002", 0.29),
                _row("3003", 0.28),
                _row("3004", 0.27),  # 業種の上限（3件）で落ちる
                _row("3005", 0.26),  # 業種の上限で落ちる
                _row("1001", 0.20),
                _row("1002", 0.19),
                _row("1003", 0.18),  # head で落ちる
                _row("1004", 0.17),  # head で落ちる
            ]
        )
        stats, record, sent = self._run(frame)
        expected = {"sector_cap": 2, "max_orders_per_run": 2}
        self.assertEqual(record["skip_reasons"], expected)
        self.assertEqual(stats["skip_reasons"], expected)
        self.assertEqual(record["skipped"], 4)
        self.assertEqual(sum(record["skip_reasons"].values()), record["skipped"])
        self.assertEqual(
            sent,
            [(s, OrderSide.BUY, 100) for s in ("3001", "3002", "3003", "1001", "1002")],
        )
        self.assertEqual(record["buy_orders"], 5)

    def test_short_head_drops_are_counted(self):
        """新規ショートの head で落ちた候補も max_orders_per_run に数える"""
        frame = pd.DataFrame(
            [_row(f"200{i}", -0.20 + 0.01 * i, diff=-0.05) for i in range(_MAX_ORDERS + 2)]
        )
        stats, record, sent = self._run(frame, short_side=True)
        self.assertEqual(record["skip_reasons"], {"max_orders_per_run": 2})
        self.assertEqual(sum(record["skip_reasons"].values()), record["skipped"])
        self.assertEqual(sent, [(f"200{i}", OrderSide.SHORT, 100) for i in range(_MAX_ORDERS)])

    def test_no_drop_has_no_cap_keys(self):
        """落ちる候補が無い朝は sector_cap も max_orders_per_run も出ない"""
        frame = pd.DataFrame([_row("3001", 0.30), _row("3002", 0.29), _row("1001", 0.20)])
        stats, record, sent = self._run(frame)
        self.assertEqual(record["skip_reasons"], {})
        self.assertEqual(record["skipped"], 0)
        self.assertEqual(len(sent), 3)


if __name__ == "__main__":
    unittest.main()
