"""
ユニットテスト: 8:50 の記録に、飛ばした候補の理由ごとの数 skip_reasons を写す（#1765）

skipped の合計は残し、skip_reasons の数の和は skipped と一致する。株数0は 16:30 の記録（#1748）と
同じ zero_qty_skip_key の鍵に分け、理由の分からないものは zero_qty にする。発注は変わらない。
"""

import json
import os
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd

from src.infrastructure.in_memory import (
    InMemoryOrderRunSink,
    InMemoryPredictionRepository,
    InMemoryTradeDiffSink,
)
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.execution import MIN_CHANGE_RATIO, run_daily_orders
from src.trading.risk_manager import RiskManager
from src.trading.types import CorrelationGateResult, TradingGateStatus
from src.utils.data_path_utils import get_results_dir

_PREDICTIONS = InMemoryPredictionRepository()
_NOW = datetime(2026, 11, 9, 8, 50, tzinfo=ZoneInfo("Asia/Tokyo"))
_ALLOWED = TradingGateStatus(
    is_allowed=True, stop_active=False, daily_loss=0.0, daily_loss_limit=15_630.0
)
_CORR_OK = CorrelationGateResult(
    is_allowed=True, enc=3.0, enc_threshold=2.0, avg_correlation=0.1, n_symbols=1
)
_SMALL = MIN_CHANGE_RATIO * 0.5
_HELD = {"symbol": "9000", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0}

# 株数を 0 にする銘柄と、そのときの risk.zero_reason
_ZERO = {
    "1001": "price<=0",
    "1002": "単元未満 投資額=50000 1単元の代金=300000 lot=100",
    "1003": None,
    "2002": "kelly<=0 kelly=-0.100",
}


def _row(symbol, score, *, diff=0.05, price=1000.0, confidence=1.0):
    return {
        "market": "jp",
        "symbol": symbol,
        "predicted_at": "20261109_083000",
        "current_price": price,
        "diff_ratio": diff,
        "confidence_ratio": confidence,
        "score": score,
    }


def _attach(df):
    out = df.copy()
    out["base_threshold"] = 0.001
    out["threshold_scale"] = 1.0
    out["effective_buy_threshold"] = 0.001
    out["effective_sell_threshold"] = -0.001
    return out


def _score(df):
    out = df.copy()
    out["multi_horizon_score"] = out["score"]
    return out


def _size(self, symbol, price, **kwargs):
    if symbol in _ZERO:
        self.zero_reason = _ZERO[symbol]
        return 0
    self.zero_reason = None
    return 100


def _record_path():
    return os.path.join(get_results_dir(), "orders", "2026-11-09.json")


class TestOrderSkipReasons(unittest.TestCase):
    def _run(self, frame, *, max_positions=2, short_side=False):
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
            patch.object(RiskManager, "calc_position_size", autospec=True, side_effect=_size),
            patch("src.trading.execution.RiskManager.update_peak_balance"),
            patch("src.trading.execution.runner.evaluate_correlation_gate", return_value=_CORR_OK),
            patch("src.trading.execution.runner._load_exit_model", return_value=None),
            patch("src.trading.execution.params.get_optimal_params", return_value={}),
            patch("src.trading.execution.runner._attach_dynamic_thresholds", side_effect=_attach),
            patch(
                "src.trading.execution.runner.apply_multi_horizon_score_column",
                side_effect=_score,
            ),
            patch(
                "src.trading.execution.selection.get_symbol_sector",
                side_effect=lambda market, symbol: f"UNKNOWN:{market}:{symbol}",
            ),
            patch("src.trading.execution.runner.MAX_POSITIONS", max_positions),
            patch("src.trading.execution.selection.MAX_ORDERS_PER_RUN", 20),
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
        sent = [(c.args[0], c.args[1]) for c in broker.send_order.call_args_list]
        return stats, record, sent

    def test_mixed_buy_skips_are_split_by_reason(self):
        """保有上限・確信度・株数0（理由3種と不明）・現在値なし・変動の小ささを分けて数える"""
        frame = pd.DataFrame(
            [
                _row("9000", -0.05, diff=_SMALL),  # 保有の売りシグナル、変動が小さい
                _row("1000", 0.20, confidence=0.01),  # 確信度の見送り
                _row("1001", 0.19),  # 株数0 price<=0
                _row("1002", 0.18),  # 株数0 単元未満
                _row("1003", 0.17),  # 株数0 理由が分からない
                _row("1004", 0.16, price=0.0),  # 現在値なし
                _row("1005", 0.15, diff=_SMALL),  # 変動が小さい
                _row("1006", 0.14),  # 買う（保有 2 で上限に届く）
                _row("1007", 0.13),  # 保有上限
                _row("1008", 0.12),  # 保有上限
            ]
        )
        stats, record, sent = self._run(frame)
        expected = {
            "sell_min_change": 1,
            "low_confidence": 1,
            "zero_qty_price": 1,
            "zero_qty_under_lot": 1,
            "zero_qty": 1,
            "no_price": 1,
            "min_change": 1,
            "max_positions": 2,
        }
        self.assertEqual(record["skip_reasons"], expected)
        self.assertEqual(record["skipped"], 9)
        self.assertEqual(sum(record["skip_reasons"].values()), record["skipped"])
        self.assertEqual(record["skipped_min_change"], 2)
        self.assertEqual(stats["skip_reasons"], expected)
        self.assertEqual(sent, [("1006", OrderSide.BUY)])
        self.assertEqual(record["buy_orders"], 1)

    def test_short_skips_use_same_keys(self):
        """新規ショートの飛ばしも同じ鍵で数え、和は skipped と一致する"""
        frame = pd.DataFrame(
            [
                _row("2000", -0.20, diff=-0.05, confidence=0.01),  # 確信度の見送り
                _row("2001", -0.19, diff=-_SMALL),  # 変動が小さい
                _row("2002", -0.18, diff=-0.05),  # 株数0 kelly<=0
                _row("2003", -0.17, diff=-0.05, price=0.0),  # 現在値なし
                _row("2004", -0.16, diff=-0.05),  # ショートする（保有 2 で上限に届く）
                _row("2005", -0.15, diff=-0.05),  # 保有上限
            ]
        )
        stats, record, sent = self._run(frame, short_side=True)
        expected = {
            "low_confidence": 1,
            "min_change": 1,
            "zero_qty_kelly": 1,
            "no_price": 1,
            "max_positions": 1,
        }
        self.assertEqual(record["skip_reasons"], expected)
        self.assertEqual(sum(record["skip_reasons"].values()), record["skipped"])
        self.assertEqual(sent, [("2004", OrderSide.SHORT)])

    def test_no_skip_records_empty_reasons(self):
        """飛ばしの無い回は空の skip_reasons と skipped 0 が入る"""
        stats, record, _ = self._run(pd.DataFrame([_row("1006", 0.14)]))
        self.assertEqual(record["skip_reasons"], {})
        self.assertEqual(record["skipped"], 0)


if __name__ == "__main__":
    unittest.main()
