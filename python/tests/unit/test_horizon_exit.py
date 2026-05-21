"""
ユニットテスト: 予測ホライズン連動保有期間管理 (Issue #259)

- _determine_entry_horizon: 最大絶対変化率のホライズンを選択
- _link_paper_order_metadata: horizon / target_exit_date を paper_orders に書き込む
- run_horizon_exit_check: 期限切れポジションを自動決済する
"""

import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd

from src.trading.execution import _determine_entry_horizon


class TestDetermineEntryHorizon(unittest.TestCase):
    def _row(self, **kwargs) -> pd.Series:
        base = {"diff_ratio": 0.01, "diff_ratio_3d": None, "diff_ratio_5d": None, "diff_ratio_10d": None}
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
        row = self._row(diff_ratio=0.01, diff_ratio_3d=0.02, diff_ratio_5d=0.03, diff_ratio_10d=0.09)
        self.assertEqual(_determine_entry_horizon(row), 10)

    def test_negative_values_use_abs(self):
        row = self._row(diff_ratio=-0.03, diff_ratio_3d=-0.01)
        self.assertEqual(_determine_entry_horizon(row), 1)

    def test_nan_ignored(self):
        row = self._row(diff_ratio=0.01, diff_ratio_3d=float("nan"), diff_ratio_5d=0.05)
        self.assertEqual(_determine_entry_horizon(row), 5)

    def test_all_equal_returns_1(self):
        row = self._row(diff_ratio=0.02, diff_ratio_3d=0.02, diff_ratio_5d=0.02, diff_ratio_10d=0.02)
        self.assertIn(_determine_entry_horizon(row), [1, 3, 5, 10])

    def test_missing_columns(self):
        row = pd.Series({"diff_ratio": 0.03})
        self.assertEqual(_determine_entry_horizon(row), 1)


class TestLinkPaperOrderMetadata(unittest.TestCase):
    @patch("src.trading.execution._db_connection")
    def test_sets_horizon_and_target_exit_date(self, mock_db):
        from src.trading.execution import _link_paper_order_metadata

        mock_con = MagicMock()
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_con)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

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

        mock_con.execute.assert_called_once()
        args = mock_con.execute.call_args[0]
        params = args[1]
        self.assertEqual(params[0], "jp")
        self.assertEqual(params[3], 3)
        self.assertEqual(params[4], target)
        self.assertEqual(params[5], "ORD-001")

    @patch("src.trading.execution._db_connection")
    def test_none_horizon_allowed(self, mock_db):
        from src.trading.execution import _link_paper_order_metadata

        mock_con = MagicMock()
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_con)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        _link_paper_order_metadata(
            order_id="ORD-002",
            market="jp",
            predicted_at="2026-05-19T10:00:00",
            signal_price=1000.0,
        )

        args = mock_con.execute.call_args[0]
        params = args[1]
        self.assertIsNone(params[3])
        self.assertIsNone(params[4])


_DB_PATCH = "src.utils.db._connection._db_connection"
_BROKER_PATCH = "src.trading.brokers.paper.paper_broker.PaperBroker"


class TestRunHorizonExitCheck(unittest.TestCase):
    @patch(_BROKER_PATCH)
    @patch(_DB_PATCH)
    def test_exits_overdue_positions(self, mock_db, mock_broker_cls):
        from src.orchestration.scheduler import run_horizon_exit_check
        from src.trading.brokers.base import OrderSide

        mock_con = MagicMock()
        mock_con.execute.return_value.fetchall.return_value = [("7203",), ("9984",)]
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_con)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

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
    @patch(_DB_PATCH)
    def test_skip_if_position_already_closed(self, mock_db, mock_broker_cls):
        from src.orchestration.scheduler import run_horizon_exit_check

        mock_con = MagicMock()
        mock_con.execute.return_value.fetchall.return_value = [("7203",)]
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_con)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

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
    @patch(_DB_PATCH)
    def test_no_symbols_does_nothing(self, mock_db, mock_broker_cls):
        from src.orchestration.scheduler import run_horizon_exit_check

        mock_con = MagicMock()
        mock_con.execute.return_value.fetchall.return_value = []
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_con)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        run_horizon_exit_check()

        mock_broker_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
