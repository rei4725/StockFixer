"""
ユニットテスト: _record_order (Issue #586)

かつて存在しない `orders` テーブルへの INSERT が _record_order 内にあり、
例外で _link_paper_order_metadata / upsert_paper_real_diff の実行が
サイレントに握りつぶされていた（paper_orders の predicted_at/signal_price/
horizon/target_exit_date が一貫して欠落し、paper_real_diff への記録も
一度も行われていなかった）。

DB を直接叩く経路は tests/unit/conftest.py の autouse `_isolate_db` フィクスチャ
経由で実 Postgres（テスト専用インスタンス）に対して検証する。
"""

import unittest

from src.trading.brokers.base import OrderSide, OrderType
from src.trading.execution.recording import _record_order
from src.utils.db._connection import _db_connection


class TestRecordOrder(unittest.TestCase):
    def setUp(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")
            con.execute("DELETE FROM paper_real_diff")
            con.execute(
                "INSERT INTO paper_orders "
                "(order_id, symbol, side, qty, order_type, status) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                ["ORD-REC-001", "7203", int(OrderSide.BUY), 100, int(OrderType.MARKET), "filled"],
            )

    def test_paper_mode_links_metadata_without_raising(self):
        """paper モードでは例外を出さず、paper_orders のメタデータが補完されること。"""
        _record_order(
            market="jp",
            predicted_at="2026-05-19T10:00:00",
            symbol="7203",
            side=OrderSide.BUY,
            qty=100,
            signal_price=1500.0,
            order_price=1502.0,
            order_type=OrderType.MARKET,
            order_result={"order_id": "ORD-REC-001", "status": "filled", "fill_price": 1502.0},
            broker=None,
            mode="paper",
            horizon=3,
        )

        with _db_connection() as con:
            row = con.execute(
                "SELECT market, predicted_at, signal_price, horizon, "
                "CAST(target_exit_date AS VARCHAR) "
                "FROM paper_orders WHERE order_id = %s",
                ["ORD-REC-001"],
            ).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], "jp")
        self.assertEqual(row[1], "2026-05-19T10:00:00")
        self.assertAlmostEqual(row[2], 1500.0)
        self.assertEqual(row[3], 3)
        self.assertIsNotNone(row[4])

    def test_paper_mode_records_paper_real_diff(self):
        """paper モードで paper_real_diff にも正しく記録されること。"""
        _record_order(
            market="jp",
            predicted_at="2026-05-19T10:00:00",
            symbol="7203",
            side=OrderSide.BUY,
            qty=100,
            signal_price=1500.0,
            order_price=1502.0,
            order_type=OrderType.MARKET,
            order_result={"order_id": "ORD-REC-001", "status": "filled", "fill_price": 1502.0},
            broker=None,
            mode="paper",
        )

        with _db_connection() as con:
            row = con.execute(
                "SELECT paper_order_id, paper_price FROM paper_real_diff "
                "WHERE market = %s AND symbol = %s AND predicted_at = %s AND side = %s",
                ["jp", "7203", "2026-05-19T10:00:00", int(OrderSide.BUY)],
            ).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], "ORD-REC-001")
        self.assertAlmostEqual(row[1], 1502.0)

    def test_live_mode_does_not_link_paper_metadata(self):
        """live モードでは paper_orders の補完（_link_paper_order_metadata）を行わないこと。"""
        _record_order(
            market="jp",
            predicted_at="2026-05-19T10:00:00",
            symbol="7203",
            side=OrderSide.BUY,
            qty=100,
            signal_price=1500.0,
            order_price=1502.0,
            order_type=OrderType.MARKET,
            order_result={"order_id": "ORD-REC-002", "status": "filled", "fill_price": 1503.0},
            broker=None,
            mode="live",
        )

        with _db_connection() as con:
            row = con.execute(
                "SELECT predicted_at FROM paper_orders WHERE order_id = %s",
                ["ORD-REC-002"],
            ).fetchone()
        self.assertIsNone(row)

        with _db_connection() as con:
            diff_row = con.execute(
                "SELECT real_order_id, real_price FROM paper_real_diff "
                "WHERE market = %s AND symbol = %s AND predicted_at = %s AND side = %s",
                ["jp", "7203", "2026-05-19T10:00:00", int(OrderSide.BUY)],
            ).fetchone()
        self.assertIsNotNone(diff_row)
        self.assertEqual(diff_row[0], "ORD-REC-002")
        self.assertAlmostEqual(diff_row[1], 1503.0)


if __name__ == "__main__":
    unittest.main()
