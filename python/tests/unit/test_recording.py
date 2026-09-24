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
from unittest.mock import MagicMock, patch

from src.infrastructure.in_memory import InMemoryTradeDiffSink
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
            trade_diff_sink=InMemoryTradeDiffSink(),
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
        """paper モードで注入された TradeDiffSink にも正しく記録されること。

        （旧実装は paper_real_diff テーブルへ直接 upsert していたが、Phase 4b で
        TradeDiffSink 経由に置き換わったため、DB 直接検証ではなく Sink の
        recorded を検証する。）
        """
        sink = InMemoryTradeDiffSink()
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
            trade_diff_sink=sink,
        )

        self.assertEqual(len(sink.recorded), 1)
        rec = sink.recorded[0]
        self.assertEqual(rec.order_id, "ORD-REC-001")
        self.assertAlmostEqual(rec.actual_price, 1502.0)

    def test_live_mode_does_not_link_paper_metadata(self):
        """live モードでは paper_orders の補完（_link_paper_order_metadata）を行わないこと。

        TradeDiffSink への記録は行うこと（Sink の recorded を検証する）。
        """
        sink = InMemoryTradeDiffSink()
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
            trade_diff_sink=sink,
        )

        with _db_connection() as con:
            row = con.execute(
                "SELECT predicted_at FROM paper_orders WHERE order_id = %s",
                ["ORD-REC-002"],
            ).fetchone()
        self.assertIsNone(row)

        self.assertEqual(len(sink.recorded), 1)
        rec = sink.recorded[0]
        self.assertEqual(rec.order_id, "ORD-REC-002")
        self.assertAlmostEqual(rec.actual_price, 1503.0)


class TestRecordOrderUsesInjectedSink(unittest.TestCase):
    """_record_order が注入された Sink へ記録すること（module-level import 撤去の回帰）。"""

    @patch("src.trading.execution.recording._link_paper_order_metadata")
    def test_records_into_injected_sink(self, _mock_link):
        from src.infrastructure.in_memory import InMemoryTradeDiffSink
        from src.trading.brokers.base import OrderSide, OrderType

        sink = InMemoryTradeDiffSink()
        _record_order(
            market="jp",
            predicted_at="2026-09-23T00:00:00",
            symbol="7203",
            side=OrderSide.BUY,
            qty=100,
            signal_price=1000.0,
            order_price=1000.0,
            order_type=OrderType.MARKET,
            order_result={"order_id": "ord-1", "fill_price": 1002.0},
            broker=MagicMock(),
            mode="paper",
            trade_diff_sink=sink,
        )

        self.assertEqual(len(sink.recorded), 1)
        rec = sink.recorded[0]
        self.assertEqual(rec.symbol, "7203")
        self.assertEqual(rec.mode, "paper")
        self.assertEqual(rec.order_id, "ord-1")
        self.assertEqual(rec.actual_price, 1002.0)

    def test_module_has_no_prediction_db_import(self):
        """src.utils.db からの関数 import が残っていないこと。"""
        import src.trading.execution.recording as mod

        self.assertFalse(hasattr(mod, "upsert_paper_real_diff"))


if __name__ == "__main__":
    unittest.main()
