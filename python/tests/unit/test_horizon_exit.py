"""
ユニットテスト: 予測ホライズン連動保有期間管理 (Issue #259)

- _determine_entry_horizon: 最大絶対変化率のホライズンを選択
- _link_paper_order_metadata: horizon / target_exit_date を paper_orders に書き込む
- run_horizon_exit_check: 期限切れポジションを自動決済する

DB を直接叩く経路（_link_paper_order_metadata・run_horizon_exit_check の
SELECT/UPDATE）は tests/unit/conftest.py の autouse `_isolate_db` フィクスチャ
経由で実 Postgres に対して検証する。Broker のモック化は DB 移行と無関係なため維持する。
"""

import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd

from src.trading.brokers.base import OrderSide
from src.trading.execution import _determine_entry_horizon
from src.utils.db._connection import _db_connection


class TestDetermineEntryHorizon(unittest.TestCase):
    def _row(self, **kwargs) -> pd.Series:
        base = {
            "diff_ratio": 0.01,
            "diff_ratio_3d": None,
            "diff_ratio_5d": None,
            "diff_ratio_10d": None,
        }
        base.update(kwargs)
        return pd.Series(base)

    def test_1d_only(self):
        row = self._row(diff_ratio=0.02)
        self.assertEqual(_determine_entry_horizon(row), 1)

    def test_3d_largest(self):
        row = self._row(diff_ratio=0.01, diff_ratio_3d=0.05, diff_ratio_5d=0.02)
        self.assertEqual(_determine_entry_horizon(row), 3)

    def test_5d_largest(self):
        row = self._row(diff_ratio=0.01, diff_ratio_3d=0.02, diff_ratio_5d=0.08)
        self.assertEqual(_determine_entry_horizon(row), 5)

    def test_10d_largest(self):
        row = self._row(
            diff_ratio=0.01, diff_ratio_3d=0.02, diff_ratio_5d=0.03, diff_ratio_10d=0.09
        )
        self.assertEqual(_determine_entry_horizon(row), 10)

    def test_negative_values_use_abs(self):
        row = self._row(diff_ratio=-0.03, diff_ratio_3d=-0.01)
        self.assertEqual(_determine_entry_horizon(row), 1)

    def test_nan_ignored(self):
        row = self._row(diff_ratio=0.01, diff_ratio_3d=float("nan"), diff_ratio_5d=0.05)
        self.assertEqual(_determine_entry_horizon(row), 5)

    def test_all_equal_returns_1(self):
        row = self._row(
            diff_ratio=0.02, diff_ratio_3d=0.02, diff_ratio_5d=0.02, diff_ratio_10d=0.02
        )
        self.assertIn(_determine_entry_horizon(row), [1, 3, 5, 10])

    def test_missing_columns(self):
        row = pd.Series({"diff_ratio": 0.03})
        self.assertEqual(_determine_entry_horizon(row), 1)


class TestLinkPaperOrderMetadata(unittest.TestCase):
    """実 Postgres に対して UPDATE paper_orders が正しく反映されることを検証する。"""

    def setUp(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")
            for order_id in ("ORD-001", "ORD-002"):
                con.execute(
                    "INSERT INTO paper_orders "
                    "(order_id, symbol, side, qty, order_type, status) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    [order_id, "7203", int(OrderSide.BUY), 100, 10, "pending"],
                )

    def test_sets_horizon_and_target_exit_date(self):
        from src.trading.execution import _link_paper_order_metadata

        today = date.today()
        target = (today + timedelta(days=3)).isoformat()

        _link_paper_order_metadata(
            order_id="ORD-001",
            market="jp",
            predicted_at="2026-05-19T10:00:00",
            signal_price=1500.0,
            horizon=3,
            target_exit_date=target,
        )

        with _db_connection() as con:
            row = con.execute(
                "SELECT market, predicted_at, signal_price, horizon, "
                "CAST(target_exit_date AS VARCHAR) "
                "FROM paper_orders WHERE order_id = %s",
                ["ORD-001"],
            ).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], "jp")
        self.assertEqual(row[1], "2026-05-19T10:00:00")
        self.assertAlmostEqual(row[2], 1500.0)
        self.assertEqual(row[3], 3)
        self.assertEqual(row[4], target)

    def test_none_horizon_allowed(self):
        from src.trading.execution import _link_paper_order_metadata

        _link_paper_order_metadata(
            order_id="ORD-002",
            market="jp",
            predicted_at="2026-05-19T10:00:00",
            signal_price=1000.0,
        )

        with _db_connection() as con:
            row = con.execute(
                "SELECT horizon, target_exit_date FROM paper_orders WHERE order_id = %s",
                ["ORD-002"],
            ).fetchone()

        self.assertIsNotNone(row)
        self.assertIsNone(row[0])
        self.assertIsNone(row[1])


_BROKER_PATCH = "src.trading.brokers.paper.paper_broker.PaperBroker"
# live モードスキップの検証専用: DB 呼び出しが一切発生しないことをアサートするための MagicMock。
# _isolate_db が注入する _test_connection とは独立した仕組み（_db_connection 関数自体を差し替える）
# なので共存できる。
_DB_PATCH = "src.utils.db._connection._db_connection"


class TestRunHorizonExitCheck(unittest.TestCase):
    def setUp(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")

    def _insert_overdue_order(self, order_id: str, symbol: str, days_overdue: int = 1) -> None:
        target_exit_date = (date.today() - timedelta(days=days_overdue)).isoformat()
        with _db_connection() as con:
            con.execute(
                "INSERT INTO paper_orders "
                "(order_id, symbol, side, qty, order_type, status, target_exit_date) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [order_id, symbol, int(OrderSide.BUY), 100, 10, "filled", target_exit_date],
            )

    @patch(_BROKER_PATCH)
    def test_exits_overdue_positions(self, mock_broker_cls):
        """実 Postgres 上で target_exit_date 超過の注文が SELECT DISTINCT symbol で
        正しく検出され、SELL 注文が発行されることを検証する（DB 層は実行に実接続を使用）。
        """
        from src.orchestration.scheduler import run_horizon_exit_check

        self._insert_overdue_order("ORD-100", "7203")
        self._insert_overdue_order("ORD-101", "9984")

        broker = MagicMock()
        broker.get_positions.return_value = [
            {"symbol": "7203", "qty": 100},
            {"symbol": "9984", "qty": 200},
        ]
        mock_broker_cls.return_value = broker

        run_horizon_exit_check()

        self.assertEqual(broker.send_order.call_count, 2)
        broker.send_order.assert_any_call("7203", OrderSide.SELL, 100)
        broker.send_order.assert_any_call("9984", OrderSide.SELL, 200)

    @patch(_BROKER_PATCH)
    def test_skip_if_position_already_closed(self, mock_broker_cls):
        from src.orchestration.scheduler import run_horizon_exit_check

        self._insert_overdue_order("ORD-102", "7203")

        broker = MagicMock()
        broker.get_positions.return_value = []
        mock_broker_cls.return_value = broker

        run_horizon_exit_check()

        broker.send_order.assert_not_called()

    @patch(_DB_PATCH)
    def test_skip_in_live_mode(self, mock_db):
        import os

        from src.orchestration.scheduler import run_horizon_exit_check

        with patch.dict(os.environ, {"AUTO_TRADE_MODE": "live"}):
            run_horizon_exit_check()

        mock_db.assert_not_called()

    @patch(_BROKER_PATCH)
    def test_no_symbols_does_nothing(self, mock_broker_cls):
        """paper_orders が空（=期限切れ注文なし）の場合、Broker は構築されない。"""
        from src.orchestration.scheduler import run_horizon_exit_check

        run_horizon_exit_check()

        mock_broker_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
