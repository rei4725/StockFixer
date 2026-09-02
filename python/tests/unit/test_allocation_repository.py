"""ユニットテスト: src.trading.allocation_strategy.repository"""

import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch


class TestGetLatestSnapshot(unittest.TestCase):
    def _mock_db(self, row):
        mock_con = MagicMock()
        mock_con.execute.return_value.fetchone.return_value = row
        return mock_con

    def test_returns_none_when_no_rows(self):
        from src.trading.allocation_strategy.repository import get_latest_snapshot

        mock_con = self._mock_db(None)
        with patch("src.trading.allocation_strategy.repository._db_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_con)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = get_latest_snapshot()
        self.assertIsNone(result)

    def test_parses_row_into_snapshot(self):
        from src.trading.allocation_strategy.repository import get_latest_snapshot

        row = (
            7,
            datetime(2026, 1, 1),
            "rebalance",
            80.0,
            85.0,
            100.0,
            50.0,
            10.0,
            120.0,
            30.0,
            5.0,
        )
        mock_con = self._mock_db(row)
        with patch("src.trading.allocation_strategy.repository._db_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_con)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = get_latest_snapshot()

        self.assertEqual(result.id, 7)
        self.assertEqual(result.executed_at, datetime(2026, 1, 1))
        self.assertEqual(result.action, "rebalance")
        self.assertEqual(result.tqqq_price, 80.0)
        self.assertEqual(result.shy_price, 85.0)
        self.assertEqual(result.tqqq_qty_before, 100.0)
        self.assertEqual(result.shy_qty_before, 50.0)
        self.assertEqual(result.cash_before, 10.0)
        self.assertEqual(result.tqqq_qty_after, 120.0)
        self.assertEqual(result.shy_qty_after, 30.0)
        self.assertEqual(result.cash_after, 5.0)


class TestListSnapshots(unittest.TestCase):
    def test_returns_empty_list_when_no_rows(self):
        from src.trading.allocation_strategy.repository import list_snapshots

        mock_con = MagicMock()
        mock_con.execute.return_value.fetchall.return_value = []
        with patch("src.trading.allocation_strategy.repository._db_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_con)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = list_snapshots()
        self.assertEqual(result, [])

    def test_returns_snapshots_in_ascending_order(self):
        from src.trading.allocation_strategy.repository import list_snapshots

        rows = [
            (1, datetime(2026, 1, 1), "initial", 80.0, 85.0, 0.0, 0.0, 100.0, 100.0, 10.0, 5.0),
            (2, datetime(2027, 1, 1), "rebalance", 90.0, 86.0, 100.0, 10.0, 5.0, 110.0, 8.0, 3.0),
        ]
        mock_con = MagicMock()
        mock_con.execute.return_value.fetchall.return_value = rows
        with patch("src.trading.allocation_strategy.repository._db_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_con)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = list_snapshots()

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].id, 1)
        self.assertEqual(result[1].id, 2)
        sql_call = mock_con.execute.call_args
        self.assertIn("ORDER BY id ASC", sql_call[0][0])


class TestInsertSnapshot(unittest.TestCase):
    def test_executes_insert_with_expected_params(self):
        from src.trading.allocation_strategy.repository import insert_snapshot

        mock_con = MagicMock()
        with patch("src.trading.allocation_strategy.repository._db_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_con)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            insert_snapshot(
                action="initial",
                tqqq_price=80.0,
                shy_price=85.0,
                tqqq_qty_before=0.0,
                shy_qty_before=0.0,
                cash_before=100_000.0,
                tqqq_qty_after=1000.0,
                shy_qty_after=235.29,
                cash_after=0.0,
            )

        mock_con.execute.assert_called_once()
        sql_call = mock_con.execute.call_args
        self.assertIn("INSERT INTO allocation_rebalance_log", sql_call[0][0])
        self.assertEqual(
            sql_call[0][1],
            ["initial", 80.0, 85.0, 0.0, 0.0, 100_000.0, 1000.0, 235.29, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
