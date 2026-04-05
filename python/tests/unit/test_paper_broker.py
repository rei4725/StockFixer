"""
ユニットテスト: PaperBroker

DuckDB を一時ファイルに差し替えてテスト。yfinance 呼び出しはモック化。
"""

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import duckdb
import pandas as pd

from src.brokers.base import OrderSide
from src.brokers.paper.paper_broker import PaperBroker

# テスト用インメモリ DB
_TEST_CON = duckdb.connect(":memory:")
_TEST_CON.execute(
    """
    CREATE TABLE IF NOT EXISTS paper_orders (
        order_id VARCHAR, market VARCHAR, predicted_at VARCHAR,
        symbol VARCHAR, side INTEGER, qty INTEGER,
        price DOUBLE, signal_price DOUBLE, order_type INTEGER, fill_price DOUBLE,
        status VARCHAR DEFAULT 'pending',
        realized_pnl DOUBLE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        filled_at TIMESTAMP
    )
    """
)
_TEST_CON.execute(
    """
    CREATE TABLE IF NOT EXISTS paper_positions (
        symbol VARCHAR PRIMARY KEY, qty INTEGER, avg_price DOUBLE,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
)
_TEST_CON.execute(
    """
    CREATE TABLE IF NOT EXISTS paper_balance (
        id INTEGER PRIMARY KEY DEFAULT 1,
        balance DOUBLE DEFAULT 1000000.0
    )
    """
)
_TEST_CON.execute("INSERT OR IGNORE INTO paper_balance (id, balance) VALUES (1, 1000000.0)")


def _get_test_con():
    return _TEST_CON


@contextmanager
def _test_db_connection():
    yield _TEST_CON


class TestPaperBrokerOrder(unittest.TestCase):
    def setUp(self):
        # 各テスト前にテーブルをリセット
        _TEST_CON.execute("DELETE FROM paper_orders")
        _TEST_CON.execute("DELETE FROM paper_positions")
        _TEST_CON.execute("UPDATE paper_balance SET balance = 1000000.0")
        self.broker = PaperBroker()

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    def test_send_order_returns_pending(self, _mock=None):
        result = self.broker.send_order("7203", OrderSide.BUY, 100)
        self.assertEqual(result["status"], "pending")
        self.assertIn("order_id", result)

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    def test_send_order_saved_to_db(self, _mock=None):
        self.broker.send_order("7203", OrderSide.BUY, 100)
        row = _TEST_CON.execute("SELECT status FROM paper_orders WHERE symbol='7203'").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "pending")

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    def test_cancel_order(self, _mock=None):
        result = self.broker.send_order("7203", OrderSide.BUY, 100)
        order_id = result["order_id"]
        cancel = self.broker.cancel_order(order_id)
        self.assertEqual(cancel["status"], "cancelled")
        row = _TEST_CON.execute(
            "SELECT status FROM paper_orders WHERE order_id=?", [order_id]
        ).fetchone()
        self.assertEqual(row[0], "cancelled")

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    def test_get_balance_initial(self, _mock=None):
        balance = self.broker.get_balance()
        self.assertEqual(balance, 1_000_000.0)

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    def test_get_positions_empty(self, _mock=None):
        positions = self.broker.get_positions()
        self.assertEqual(positions, [])


class TestPaperBrokerSettle(unittest.TestCase):
    def setUp(self):
        _TEST_CON.execute("DELETE FROM paper_orders")
        _TEST_CON.execute("DELETE FROM paper_positions")
        _TEST_CON.execute("UPDATE paper_balance SET balance = 1000000.0")
        self.broker = PaperBroker()

    def _mock_yf_download(self, *args, **kwargs):
        return pd.DataFrame(
            {"Open": [1000.0], "High": [1050.0], "Low": [990.0], "Close": [1020.0]},
            index=pd.to_datetime(["2026-03-15"]),
        )

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    @patch("src.brokers.paper.paper_broker.yf_client.download")
    def test_settle_market_buy(self, mock_yf, _mock_con=None):
        mock_yf.return_value = self._mock_yf_download()
        self.broker.send_order("7203", OrderSide.BUY, 100)
        settled = self.broker.settle_pending_orders()
        self.assertEqual(len(settled), 1)
        self.assertAlmostEqual(settled[0]["fill_price"], 1000.0)
        # 残高が減少していること
        balance = _TEST_CON.execute("SELECT balance FROM paper_balance").fetchone()[0]
        self.assertAlmostEqual(balance, 1_000_000.0 - 1000.0 * 100)

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    @patch("src.brokers.paper.paper_broker.yf_client.download")
    def test_settle_creates_position(self, mock_yf, _mock_con=None):
        mock_yf.return_value = self._mock_yf_download()
        self.broker.send_order("7203", OrderSide.BUY, 100)
        self.broker.settle_pending_orders()
        pos = _TEST_CON.execute(
            "SELECT qty, avg_price FROM paper_positions WHERE symbol='7203'"
        ).fetchone()
        self.assertIsNotNone(pos)
        self.assertEqual(pos[0], 100)

    @patch("src.brokers.paper.paper_broker.upsert_paper_real_diff")
    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    @patch("src.brokers.paper.paper_broker.yf_client.download")
    def test_settle_updates_paper_real_diff(self, mock_yf, mock_upsert):
        mock_yf.return_value = self._mock_yf_download()
        result = self.broker.send_order("7203", OrderSide.BUY, 100)
        _TEST_CON.execute(
            "UPDATE paper_orders SET market='jp', predicted_at='20260405_085000', "
            "signal_price=995.0 WHERE order_id=?",
            [result["order_id"]],
        )

        self.broker.settle_pending_orders()

        mock_upsert.assert_called_once()
        self.assertEqual(mock_upsert.call_args.kwargs["market"], "jp")
        self.assertEqual(mock_upsert.call_args.kwargs["symbol"], "7203")
        self.assertAlmostEqual(mock_upsert.call_args.kwargs["actual_price"], 1000.0)


if __name__ == "__main__":
    unittest.main()
