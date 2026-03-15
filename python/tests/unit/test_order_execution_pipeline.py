"""
ユニットテスト: OrderExecutionPipeline

DuckDB・Broker・yfinance はすべて MagicMock で差し替える。
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.brokers.base import BrokerBase
from src.services.order_execution_pipeline import (
    BUY_THRESHOLD,
    MAX_ORDERS_PER_RUN,
    run_daily_orders,
)


def _make_broker(balance=1_000_000.0, positions=None):
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_balance.return_value = balance
    broker.get_positions.return_value = positions or []
    broker.send_order.return_value = {"order_id": "test-001", "status": "pending"}
    return broker


def _make_predictions(n_buy=3, n_sell=0):
    """テスト用予測 DataFrame を生成する"""
    rows = []
    for i in range(n_buy):
        rows.append(
            {
                "market": "jp",
                "symbol": f"720{i}",
                "current_price": 1000.0 + i * 10,
                "diff_ratio": BUY_THRESHOLD + 0.01 * (i + 1),
            }
        )
    for j in range(n_sell):
        rows.append(
            {
                "market": "jp",
                "symbol": f"900{j}",
                "current_price": 800.0,
                "diff_ratio": -0.02,
            }
        )
    return pd.DataFrame(rows)


class TestRunDailyOrders(unittest.TestCase):
    def _patch_pipeline(self, predictions, risk_allowed=True, calc_qty=100):
        """共通パッチャー群をまとめて返す"""
        patches = [
            patch(
                "src.services.order_execution_pipeline._load_latest_predictions",
                return_value=predictions,
            ),
            patch(
                "src.services.order_execution_pipeline._record_order",
            ),
            patch(
                "src.services.risk_manager.RiskManager.is_trading_allowed",
                return_value=risk_allowed,
            ),
            patch(
                "src.services.risk_manager.RiskManager.calc_position_size",
                return_value=calc_qty,
            ),
            patch(
                "src.services.risk_manager.RiskManager._get_daily_realized_loss",
                return_value=0.0,
            ),
            patch(
                "src.services.risk_manager.RiskManager._get_consecutive_losses",
                return_value=0,
            ),
        ]
        return patches

    def _start_patches(self, patches):
        mocks = [p.start() for p in patches]
        return mocks, patches

    def _stop_patches(self, patches):
        for p in patches:
            p.stop()

    def test_buy_orders_placed(self):
        broker = _make_broker()
        predictions = _make_predictions(n_buy=3)
        patches = self._patch_pipeline(predictions)
        _, patch_list = self._start_patches(patches)
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertGreater(stats["buy_orders"], 0)
            self.assertEqual(stats["errors"], 0)
        finally:
            self._stop_patches(patch_list)

    def test_risk_blocked_no_orders(self):
        broker = _make_broker()
        predictions = _make_predictions(n_buy=3)
        patches = self._patch_pipeline(predictions, risk_allowed=False)
        _, patch_list = self._start_patches(patches)
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertEqual(stats["buy_orders"], 0)
            broker.send_order.assert_not_called()
        finally:
            self._stop_patches(patch_list)

    def test_empty_predictions_no_orders(self):
        broker = _make_broker()
        patches = self._patch_pipeline(pd.DataFrame())
        _, patch_list = self._start_patches(patches)
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertEqual(stats["buy_orders"], 0)
        finally:
            self._stop_patches(patch_list)

    def test_max_orders_per_run_respected(self):
        broker = _make_broker()
        # MAX_ORDERS_PER_RUN より多い銘柄を提供
        predictions = _make_predictions(n_buy=MAX_ORDERS_PER_RUN + 3)
        patches = self._patch_pipeline(predictions)
        _, patch_list = self._start_patches(patches)
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertLessEqual(stats["buy_orders"], MAX_ORDERS_PER_RUN)
        finally:
            self._stop_patches(patch_list)

    def test_held_symbol_not_rebought(self):
        # 既に保有している銘柄には買い注文を出さない
        positions = [{"symbol": "7200", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0}]
        broker = _make_broker(positions=positions)
        predictions = _make_predictions(n_buy=1)  # symbol="7200"
        patches = self._patch_pipeline(predictions)
        _, patch_list = self._start_patches(patches)
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertEqual(stats["buy_orders"], 0)
        finally:
            self._stop_patches(patch_list)

    def test_zero_qty_skipped(self):
        broker = _make_broker()
        predictions = _make_predictions(n_buy=2)
        patches = self._patch_pipeline(predictions, calc_qty=0)
        _, patch_list = self._start_patches(patches)
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper")
            self.assertEqual(stats["buy_orders"], 0)
            self.assertGreater(stats["skipped"], 0)
        finally:
            self._stop_patches(patch_list)


if __name__ == "__main__":
    unittest.main()
