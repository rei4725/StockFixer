"""ユニットテスト: PostgresOrderRunSink（order_run_summary テーブルへの書き込み）。"""

import unittest

from src.domain.ports import OrderRunSink
from src.domain.types import OrderRunSummary
from src.infrastructure.persistence.order_run_repository import PostgresOrderRunSink
from src.utils.db._connection import _db_connection


class TestPostgresOrderRunSink(unittest.TestCase):
    def setUp(self):
        with _db_connection() as con:
            con.execute("DELETE FROM order_run_summary")

    def test_implements_port(self):
        self.assertIsInstance(PostgresOrderRunSink(), OrderRunSink)

    def test_save_inserts_one_row(self):
        PostgresOrderRunSink().save(
            OrderRunSummary(
                run_id="run-adapter-1",
                market="jp",
                mode="paper",
                buy_orders=3,
                sell_orders=1,
                short_orders=2,
                skipped=4,
                skipped_min_change=5,
                total_turnover=987654.0,
                min_change_ratio=0.005,
            )
        )

        with _db_connection() as con:
            row = con.execute(
                "SELECT market, mode, buy_orders, sell_orders, short_orders, "
                "skipped, skipped_min_change, total_turnover, min_change_ratio "
                "FROM order_run_summary WHERE run_id = %s",
                ["run-adapter-1"],
            ).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], "jp")
        self.assertEqual(row[1], "paper")
        self.assertEqual(row[2], 3)
        self.assertEqual(row[3], 1)
        self.assertEqual(row[4], 2)
        self.assertEqual(row[5], 4)
        self.assertEqual(row[6], 5)
        self.assertAlmostEqual(float(row[7]), 987654.0, places=3)
        self.assertAlmostEqual(float(row[8]), 0.005, places=6)
