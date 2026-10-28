"""
ユニットテスト: 8:50 の道が、最新の予想の回に無い銘柄の古い行で買わない（#1304）

ウォッチリストから外れた銘柄や予想を飛ばした銘柄の行は prediction_results に残る。
古い行の diff_ratio が買いの閾値を超えても、その銘柄に BUY を出さない。
"""

import unittest
from unittest.mock import patch

from tests.unit.test_order_execution_pipeline import _make_broker

from config.settings import BUY_THRESHOLD
from src.infrastructure.in_memory import InMemoryOrderRunSink, InMemoryTradeDiffSink
from src.infrastructure.persistence.prediction_result_repository import (
    PostgresPredictionResultRepository,
)
from src.prediction.db import save_prediction_results
from src.prediction.types import PredictionResult
from src.trading.brokers.base import OrderType
from src.trading.execution import run_daily_orders
from src.trading.types import TradingGateStatus
from src.utils.db._connection import _db_connection


def _run(broker):
    gate = TradingGateStatus(is_allowed=True, stop_active=False, reason_code=None, reason=None)
    with (
        patch("src.trading.execution.runner._record_order"),
        patch(
            "src.trading.execution.runner._choose_order_params",
            return_value=(OrderType.MARKET, 0.0, "market", "open"),
        ),
        patch("src.trading.execution.RiskManager.evaluate_trading_gate", return_value=gate),
        patch("src.trading.execution.RiskManager.calc_position_size", return_value=100),
        patch("src.trading.risk_manager.RiskManager._get_daily_realized_loss", return_value=0.0),
        patch("src.trading.risk_manager.RiskManager._get_consecutive_losses", return_value=0),
        patch(
            "src.trading.execution.selection.get_symbol_sector",
            side_effect=lambda market, symbol: f"UNKNOWN:{market}:{symbol}",
        ),
        patch("src.trading.execution.runner._load_pending_exits") as pending_exits,
    ):
        pending_exits.return_value.has.return_value = False
        return run_daily_orders(
            broker,
            order_run_sink=InMemoryOrderRunSink(),
            trade_diff_sink=InMemoryTradeDiffSink(),
            prediction_repo=PostgresPredictionResultRepository(),
            market="jp",
            mode="paper",
        )


class TestOrderLatestRunOnly(unittest.TestCase):
    def setUp(self):
        with _db_connection() as con:
            con.execute("DELETE FROM prediction_results")

    def tearDown(self):
        with _db_connection() as con:
            con.execute("DELETE FROM prediction_results")

    def test_stale_row_above_buy_threshold_is_not_bought(self):
        """前の回にしか無い銘柄は、diff_ratio が閾値を超えても買わない。"""
        save_prediction_results(
            "20261019_224246", [PredictionResult("jp", "5202", 500.0, 600.0, 0.20, 2)]
        )
        save_prediction_results(
            "20261027_224340",
            [PredictionResult("jp", "7200", 1000.0, 1015.0, BUY_THRESHOLD + 0.01, 2)],
        )
        broker = _make_broker()
        broker.get_pending_buy_symbols = lambda: set()

        _run(broker)

        bought = {c.args[0] for c in broker.send_order.call_args_list}
        self.assertNotIn("5202", bought)
        self.assertEqual(bought, {"7200"})


if __name__ == "__main__":
    unittest.main()
