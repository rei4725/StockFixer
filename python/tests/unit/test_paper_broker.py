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
_TEST_CON.execute(
    """
    CREATE TABLE IF NOT EXISTS paper_short_positions (
        symbol          VARCHAR NOT NULL PRIMARY KEY,
        qty             INTEGER NOT NULL,
        avg_short_price DOUBLE NOT NULL,
        unrealized_pnl  DOUBLE,
        opened_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """
)


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


class TestPaperBrokerShort(unittest.TestCase):
    def setUp(self):
        _TEST_CON.execute("DELETE FROM paper_orders")
        _TEST_CON.execute("DELETE FROM paper_positions")
        _TEST_CON.execute("DELETE FROM paper_short_positions")
        _TEST_CON.execute("UPDATE paper_balance SET balance = 1000000.0")
        self.broker = PaperBroker()

    def _mock_yf_download(self):
        return pd.DataFrame(
            {"Open": [1000.0], "High": [1050.0], "Low": [990.0], "Close": [1020.0]},
            index=pd.to_datetime(["2026-03-15"]),
        )

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    def test_send_short_order_returns_pending(self):
        result = self.broker.send_order("7203", OrderSide.SHORT, 100, price=1500.0)
        self.assertEqual(result["status"], "pending")
        self.assertIn("order_id", result)

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    def test_send_short_order_saved_with_side3(self):
        self.broker.send_order("7203", OrderSide.SHORT, 100, price=1500.0)
        row = _TEST_CON.execute("SELECT side FROM paper_orders WHERE symbol='7203'").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 3)

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    def test_send_short_order_creates_short_position(self):
        """send_order(SHORT) で paper_short_positions に即時仮登録されること"""
        self.broker.send_order("7203", OrderSide.SHORT, 100, price=1500.0)
        pos = _TEST_CON.execute(
            "SELECT qty, avg_short_price FROM paper_short_positions WHERE symbol='7203'"
        ).fetchone()
        self.assertIsNotNone(pos)
        self.assertEqual(pos[0], 100)
        self.assertAlmostEqual(pos[1], 1500.0)

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    def test_send_short_order_weighted_avg_update(self):
        """既存ポジションがあれば加重平均単価が更新されること"""
        _TEST_CON.execute(
            "INSERT INTO paper_short_positions"
            " (symbol, qty, avg_short_price) VALUES ('7203', 100, 1200.0)"
        )
        self.broker.send_order("7203", OrderSide.SHORT, 100, price=1400.0)
        pos = _TEST_CON.execute(
            "SELECT qty, avg_short_price FROM paper_short_positions WHERE symbol='7203'"
        ).fetchone()
        self.assertEqual(pos[0], 200)
        self.assertAlmostEqual(pos[1], 1300.0)  # (1200*100 + 1400*100) / 200

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    @patch("src.brokers.paper.paper_broker.yf_client.download")
    def test_settle_short_updates_position_to_fill_price(self, mock_yf):
        """settle 後に paper_short_positions が実際の約定値段で更新されること"""
        mock_yf.return_value = self._mock_yf_download()
        # send_order で仮登録（price=1500）
        self.broker.send_order("7203", OrderSide.SHORT, 100, price=1500.0)
        settled = self.broker.settle_pending_orders()
        self.assertEqual(len(settled), 1)
        self.assertAlmostEqual(settled[0]["fill_price"], 1000.0)
        # settle 後は fill_price (open=1000) で上書きされること
        pos = _TEST_CON.execute(
            "SELECT qty, avg_short_price FROM paper_short_positions WHERE symbol='7203'"
        ).fetchone()
        self.assertIsNotNone(pos)
        self.assertEqual(pos[0], 100)
        self.assertAlmostEqual(pos[1], 1000.0)

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    @patch("src.brokers.paper.paper_broker.yf_client.download")
    def test_settle_short_cover_computes_realized_pnl(self, mock_yf):
        """SHORT_COVER の約定で realized_pnl が正しく計算されること"""
        mock_yf.return_value = self._mock_yf_download()
        # 空売りポジション (avg=1200) を用意
        _TEST_CON.execute(
            "INSERT INTO paper_short_positions"
            " (symbol, qty, avg_short_price) VALUES ('7203', 100, 1200.0)"
        )
        result = self.broker.send_order("7203", OrderSide.SHORT_COVER, 100)
        self.broker.settle_pending_orders()
        row = _TEST_CON.execute(
            "SELECT realized_pnl FROM paper_orders WHERE order_id=?", [result["order_id"]]
        ).fetchone()
        self.assertIsNotNone(row)
        # realized_pnl = (avg_short_price - fill_price) * qty = (1200 - 1000) * 100
        self.assertAlmostEqual(row[0], 20000.0)

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    @patch("src.brokers.paper.paper_broker.yf_client.download")
    def test_settle_short_cover_reduces_position(self, mock_yf):
        """SHORT_COVER 約定後に paper_short_positions の qty が減少すること"""
        mock_yf.return_value = self._mock_yf_download()
        _TEST_CON.execute(
            "INSERT INTO paper_short_positions"
            " (symbol, qty, avg_short_price) VALUES ('7203', 200, 1200.0)"
        )
        self.broker.send_order("7203", OrderSide.SHORT_COVER, 100)
        self.broker.settle_pending_orders()
        pos = _TEST_CON.execute(
            "SELECT qty FROM paper_short_positions WHERE symbol='7203'"
        ).fetchone()
        self.assertIsNotNone(pos)
        self.assertEqual(pos[0], 100)

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    @patch("src.brokers.paper.paper_broker.yf_client.download")
    def test_settle_short_cover_full_removes_position(self, mock_yf):
        """全数量返済で paper_short_positions レコードが削除されること"""
        mock_yf.return_value = self._mock_yf_download()
        _TEST_CON.execute(
            "INSERT INTO paper_short_positions"
            " (symbol, qty, avg_short_price) VALUES ('7203', 100, 1200.0)"
        )
        self.broker.send_order("7203", OrderSide.SHORT_COVER, 100)
        self.broker.settle_pending_orders()
        pos = _TEST_CON.execute(
            "SELECT qty FROM paper_short_positions WHERE symbol='7203'"
        ).fetchone()
        self.assertIsNone(pos)

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    @patch("src.brokers.paper.paper_broker.yf_client.download")
    def test_get_short_positions(self, mock_yf):
        """get_short_positions が paper_short_positions を正しく返すこと"""
        mock_yf.return_value = self._mock_yf_download()
        _TEST_CON.execute(
            "INSERT INTO paper_short_positions"
            " (symbol, qty, avg_short_price) VALUES ('7203', 100, 1200.0)"
        )
        positions = self.broker.get_short_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["symbol"], "7203")
        self.assertEqual(positions[0]["qty"], 100)
        self.assertAlmostEqual(positions[0]["avg_short_price"], 1200.0)
        self.assertIn("unrealized_pnl", positions[0])

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    def test_get_short_positions_empty(self):
        positions = self.broker.get_short_positions()
        self.assertEqual(positions, [])


class TestPaperBrokerGetToken(unittest.TestCase):
    def setUp(self):
        self.broker = PaperBroker()

    def test_get_token_returns_paper_mode(self):
        self.assertEqual(self.broker.get_token(), "paper_mode")

    def test_get_token_returns_string(self):
        self.assertIsInstance(self.broker.get_token(), str)


class TestPaperBrokerGetBalance(unittest.TestCase):
    def setUp(self):
        _TEST_CON.execute("UPDATE paper_balance SET balance = 1000000.0")
        self.broker = PaperBroker()

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    def test_get_balance_returns_current_value(self):
        _TEST_CON.execute("UPDATE paper_balance SET balance = 500000.0")
        balance = self.broker.get_balance()
        self.assertAlmostEqual(balance, 500000.0)
        _TEST_CON.execute("UPDATE paper_balance SET balance = 1000000.0")

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    def test_get_balance_returns_initial_when_table_empty(self):
        _TEST_CON.execute("DELETE FROM paper_balance")
        balance = self.broker.get_balance()
        from config.settings import PAPER_INITIAL_BALANCE

        self.assertAlmostEqual(balance, PAPER_INITIAL_BALANCE)
        _TEST_CON.execute("INSERT INTO paper_balance (id, balance) VALUES (1, 1000000.0)")


class TestPaperBrokerGetOrders(unittest.TestCase):
    def setUp(self):
        _TEST_CON.execute("DELETE FROM paper_orders")
        _TEST_CON.execute("DELETE FROM paper_positions")
        _TEST_CON.execute("UPDATE paper_balance SET balance = 1000000.0")
        self.broker = PaperBroker()

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    def test_get_orders_returns_list(self):
        result = self.broker.get_orders()
        self.assertIsInstance(result, list)

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    def test_get_orders_empty_when_no_orders(self):
        result = self.broker.get_orders()
        self.assertEqual(result, [])

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    def test_get_orders_contains_todays_order(self):
        self.broker.send_order("7203", OrderSide.BUY, 100, price=1000.0)
        orders = self.broker.get_orders()
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["symbol"], "7203")
        self.assertEqual(orders[0]["side"], int(OrderSide.BUY))
        self.assertEqual(orders[0]["status"], "pending")

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    def test_get_orders_includes_required_keys(self):
        self.broker.send_order("9984", OrderSide.SELL, 50, price=2000.0)
        orders = self.broker.get_orders()
        self.assertEqual(len(orders), 1)
        for key in ("order_id", "symbol", "side", "qty", "price", "status"):
            self.assertIn(key, orders[0])
        self.assertEqual(orders[0]["qty"], 50)


class TestPaperBrokerGetPositionsAdditional(unittest.TestCase):
    def setUp(self):
        _TEST_CON.execute("DELETE FROM paper_orders")
        _TEST_CON.execute("DELETE FROM paper_positions")
        _TEST_CON.execute("UPDATE paper_balance SET balance = 1000000.0")
        self.broker = PaperBroker()

    def _mock_df(self):
        return pd.DataFrame(
            {"Open": [1000.0], "High": [1050.0], "Low": [990.0], "Close": [1020.0]},
            index=pd.to_datetime(["2026-03-15"]),
        )

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    @patch("src.brokers.paper.paper_broker.yf_client.download")
    def test_get_positions_returns_position_with_pnl(self, mock_yf):
        mock_yf.return_value = self._mock_df()
        _TEST_CON.execute(
            "INSERT INTO paper_positions (symbol, qty, avg_price) VALUES ('7203', 100, 1000.0)"
        )
        positions = self.broker.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["symbol"], "7203")
        self.assertEqual(positions[0]["qty"], 100)
        self.assertAlmostEqual(positions[0]["avg_price"], 1000.0)
        self.assertIn("unrealized_pnl", positions[0])
        self.assertAlmostEqual(positions[0]["unrealized_pnl"], 2000.0)

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    @patch("src.brokers.paper.paper_broker.yf_client.download")
    def test_get_positions_fallback_on_yf_error(self, mock_yf):
        mock_yf.side_effect = Exception("yfinance error")
        _TEST_CON.execute(
            "INSERT INTO paper_positions (symbol, qty, avg_price) VALUES ('7203', 100, 1000.0)"
        )
        positions = self.broker.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertAlmostEqual(positions[0]["unrealized_pnl"], 0.0)


class TestPaperBrokerGetPnlSummary(unittest.TestCase):
    def setUp(self):
        _TEST_CON.execute("DELETE FROM paper_orders")
        _TEST_CON.execute("DELETE FROM paper_positions")
        _TEST_CON.execute("UPDATE paper_balance SET balance = 1000000.0")
        self.broker = PaperBroker()

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    def test_get_pnl_summary_returns_dict(self):
        result = self.broker.get_pnl_summary()
        self.assertIsInstance(result, dict)

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    def test_get_pnl_summary_zero_when_no_trades(self):
        result = self.broker.get_pnl_summary()
        self.assertAlmostEqual(result["realized_pnl"], 0.0)
        self.assertEqual(result["trade_count"], 0)
        self.assertIsNone(result["started_at"])

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    def test_get_pnl_summary_contains_balance(self):
        result = self.broker.get_pnl_summary()
        self.assertAlmostEqual(result["balance"], 1000000.0)
        from config.settings import PAPER_INITIAL_BALANCE

        self.assertAlmostEqual(result["initial_balance"], PAPER_INITIAL_BALANCE)

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    def test_get_pnl_summary_required_keys(self):
        result = self.broker.get_pnl_summary()
        for key in (
            "realized_pnl",
            "unrealized_pnl",
            "total_pnl",
            "balance",
            "initial_balance",
            "trade_count",
            "started_at",
        ):
            self.assertIn(key, result)

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    def test_get_pnl_summary_with_filled_sell_order(self):
        _TEST_CON.execute(
            """
            INSERT INTO paper_orders
                (order_id, symbol, side, qty, price, order_type,
                 status, realized_pnl, filled_at)
            VALUES ('test_pnl_01', '7203', 2, 100, 1000.0, 10,
                    'filled', 5000.0, CURRENT_TIMESTAMP)
            """
        )
        result = self.broker.get_pnl_summary()
        self.assertAlmostEqual(result["realized_pnl"], 5000.0)
        self.assertEqual(result["trade_count"], 1)
        self.assertIsNotNone(result["started_at"])

    @patch("src.brokers.paper.paper_broker._db_connection", new=_test_db_connection)
    def test_get_pnl_summary_total_pnl_equals_realized_plus_unrealized(self):
        result = self.broker.get_pnl_summary()
        self.assertAlmostEqual(
            result["total_pnl"],
            result["realized_pnl"] + result["unrealized_pnl"],
        )


if __name__ == "__main__":
    unittest.main()
