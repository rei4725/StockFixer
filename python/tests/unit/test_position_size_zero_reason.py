"""ユニットテスト: 発注株数が 0 になった理由を分ける（#1734）

8:50 の予想の道で、新規買いの候補が毎朝「発注株数 0 → スキップ（残高不足or上限）」の
同じ1行で消え、Kelly がマイナスなのか、1単元に届かないのか、縮小で単元を割ったのかが
読めなかった。calc_position_size は 0 の理由を zero_reason に残し、INFO で出す。
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.infrastructure.in_memory import (
    InMemoryOrderRunSink,
    InMemoryPredictionRepository,
    InMemoryTradeDiffSink,
)
from src.trading.brokers.base import BrokerBase, OrderType
from src.trading.execution import run_daily_orders
from src.trading.position_size_reason import zero_size_reason
from src.trading.risk_manager import RiskManager
from src.trading.types import TradingGateStatus
from src.utils.db._connection import _db_connection

_RISK_LOGGER = "src.trading.position_size_reason"
_RUNNER_LOGGER = "src.trading.execution.runner"


@pytest.fixture(autouse=True)
def _no_real_vix_fetch():
    with patch("yfinance.Ticker", side_effect=RuntimeError("unit テストでは ^VIX を読まない")):
        yield


def _broker(balance: float) -> MagicMock:
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_balance.return_value = balance
    broker.get_positions.return_value = []
    broker.get_short_positions.return_value = []
    broker.send_order.return_value = {"order_id": "ord-1734", "status": "pending"}
    return broker


def _risk(balance: float, market: str = "jp") -> RiskManager:
    risk = RiskManager(_broker(balance), market=market)
    risk.get_current_dd_ratio = MagicMock(return_value=0.0)
    risk._vix_cache = MagicMock()
    risk._vix_cache.get.return_value = 15.0  # 縮小の掛からない平時の VIX
    return risk


class TestCalcPositionSizeZeroReason(unittest.TestCase):
    def test_price_not_positive(self):
        risk = _risk(2_000_000.0)
        with self.assertLogs(_RISK_LOGGER, level="INFO") as logs:
            qty = risk.calc_position_size("7203", price=0.0)
        self.assertEqual(qty, 0)
        self.assertEqual(risk.zero_reason, "price<=0")
        self.assertIn("7203: 発注株数 0 の理由 price<=0", logs.output[-1])

    def test_kelly_not_positive(self):
        """11-04 の 6525 と同じ BT の値。期待値がマイナスで Kelly が 0 に切られる。"""
        risk = _risk(761_441.0)
        with self.assertLogs(_RISK_LOGGER, level="INFO") as logs:
            qty = risk.calc_position_size(
                "6525", 5057.0, win_rate=0.372, avg_win=0.091, avg_loss=0.061
            )
        self.assertEqual(qty, 0)
        self.assertTrue(risk.zero_reason.startswith("kelly<=0"), risk.zero_reason)
        self.assertIn("kelly<=0", logs.output[-1])

    def test_under_one_lot(self):
        """11-04 の 6532 と同じ値。投資額が1単元の代金に届かない。"""
        risk = _risk(761_441.0)
        with self.assertLogs(_RISK_LOGGER, level="INFO"):
            qty = risk.calc_position_size(
                "6532", 11_735.0, win_rate=0.519, avg_win=0.104, avg_loss=0.063
            )
        self.assertEqual(qty, 0)
        reason = risk.zero_reason
        self.assertTrue(reason.startswith("単元未満"), reason)
        self.assertIn("1単元の代金=1173500", reason)
        self.assertIn("lot=100", reason)
        self.assertIn("投資額=", reason)

    def test_scaled_below_one_lot(self):
        """縮小の前は単元に届くが、信頼度の縮小で単元を割る。"""
        risk = _risk(2_000_000.0)
        self.assertGreater(risk.calc_position_size("7203", 1000.0), 0)
        with self.assertLogs(_RISK_LOGGER, level="INFO"):
            qty = risk.calc_position_size("7203", 1000.0, confidence_ratio=0.01)
        self.assertEqual(qty, 0)
        reason = risk.zero_reason
        self.assertTrue(reason.startswith("縮小で単元割れ"), reason)
        self.assertIn("confidence=0.01", reason)
        self.assertNotIn("vix=", reason)
        self.assertNotIn("dd=", reason)

    def test_reason_cleared_when_qty_positive(self):
        risk = _risk(2_000_000.0)
        risk.calc_position_size("7203", price=0.0)
        qty = risk.calc_position_size("7203", 1000.0)
        self.assertGreater(qty, 0)
        self.assertIsNone(risk.zero_reason)


class TestZeroSizeReasonOrder(unittest.TestCase):
    def test_price_wins_over_kelly(self):
        self.assertEqual(zero_size_reason(0.0, 0.0, 0.0, 0, 100, {}), "price<=0")

    def test_scaled_lists_only_shrinking_scales(self):
        scales = {"confidence": 1.0, "dd": 0.5, "vix": 0.25}
        reason = zero_size_reason(0.3, 150_000.0, 1000.0, 150, 100, scales)
        self.assertEqual(reason, "縮小で単元割れ base_qty=150 lot=100 dd=0.50 vix=0.25")


_PREDICTIONS = InMemoryPredictionRepository()


class TestRunnerLogsZeroReason(unittest.TestCase):
    """予想の道の買いとショートの両方で、0株の行に理由が出る。"""

    def setUp(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")

    def _run(self, diff_ratio: float, reason: str):
        def _zero(self, symbol, price, **kwargs):
            self.zero_reason = reason
            return 0

        predictions = pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "6532",
                    "current_price": 800.0,
                    "diff_ratio": diff_ratio,
                    "confidence_ratio": 1.0,
                }
            ]
        )
        gate = TradingGateStatus(
            is_allowed=True,
            stop_active=False,
            reason_code=None,
            reason=None,
            daily_loss=0.0,
            daily_loss_limit=None,
        )
        corr = MagicMock(is_allowed=True, enc=0.0, avg_correlation=0.0, reason=None)
        patches = [
            patch.object(_PREDICTIONS, "get_latest_by_market", return_value=predictions),
            patch("src.trading.execution.runner._record_order"),
            patch(
                "src.trading.execution.runner._choose_order_params",
                return_value=(OrderType.MARKET, 0.0, "market", "open"),
            ),
            patch("src.trading.execution.RiskManager.evaluate_trading_gate", return_value=gate),
            patch(
                "src.trading.execution.RiskManager.calc_position_size",
                autospec=True,
                side_effect=_zero,
            ),
            patch("src.trading.execution.sl_tp.RiskManager"),
            patch("src.trading.execution.runner.evaluate_correlation_gate", return_value=corr),
            patch("src.trading.execution.runner.ENABLE_SHORT_SIDE", True),
            patch(
                "src.trading.execution.selection.get_symbol_sector",
                side_effect=lambda market, symbol: f"UNKNOWN:{market}:{symbol}",
            ),
        ]
        for p in patches:
            started = p.start()
            self.addCleanup(p.stop)
            if p.attribute == "RiskManager":
                started.return_value.check_sl_tp.return_value = (False, False, "")
        broker = _broker(1_000_000.0)
        with self.assertLogs(_RUNNER_LOGGER, level="INFO") as logs:
            stats = run_daily_orders(
                broker,
                order_run_sink=InMemoryOrderRunSink(),
                trade_diff_sink=InMemoryTradeDiffSink(),
                prediction_repo=_PREDICTIONS,
                market="jp",
                mode="paper",
            )
        broker.send_order.assert_not_called()
        return stats, [line for line in logs.output if "発注株数 0" in line]

    def test_buy_zero_line_has_reason(self):
        stats, lines = self._run(0.05, "kelly<=0 kelly=0.000")
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("6532: 発注株数 0 → スキップ（kelly<=0 kelly=0.000）", lines[0])
        self.assertEqual(stats["buy_orders"], 0)

    def test_short_zero_line_has_reason(self):
        stats, lines = self._run(-0.05, "単元未満 投資額=64500 1単元の代金=80000 lot=100")
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("6532: 発注株数 0 → ショートせず（単元未満 投資額=64500", lines[0])
        self.assertEqual(stats["short_orders"], 0)


if __name__ == "__main__":
    unittest.main()
