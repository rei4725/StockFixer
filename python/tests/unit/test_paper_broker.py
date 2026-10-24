"""
ユニットテスト: PaperBroker

実 Postgres（tests/unit/conftest.py の autouse `_isolate_db` フィクスチャによる
トランザクションロールバック分離）を経由してテストする。yfinance 呼び出しはモック化。
"""

import math
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.domain.ports import MarketDataPort
from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.trading.brokers.base import OrderSide, OrderType
from src.trading.brokers.paper import paper_broker
from src.trading.brokers.paper.paper_broker import PaperBroker
from src.utils.db._connection import _db_connection

_TODAY = date(2026, 3, 15)


@pytest.fixture(autouse=True)
def _fixed_today(monkeypatch):
    """約定のジョブの「今日」（JST）を、模擬の OHLCV の最後の行の日に合わせる（#805）"""
    monkeypatch.setattr(paper_broker, "_today_jst", lambda: _TODAY)


def _make_mock_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        {"Open": [1000.0], "High": [1050.0], "Low": [990.0], "Close": [1020.0]},
        index=pd.to_datetime(["2026-03-15"]),
    )


def _make_market_data_mock() -> MagicMock:
    mock = MagicMock(spec=MarketDataPort)
    mock.get_ohlcv.return_value = _make_mock_ohlcv()
    return mock


class TestPaperBrokerOrder(unittest.TestCase):
    def setUp(self):
        # 各テスト前にテーブルをリセット（_isolate_db がテスト終了時にロールバックする）
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")
            con.execute("DELETE FROM paper_positions")
            con.execute("UPDATE paper_balance SET balance = 1000000.0")
        self.broker = PaperBroker(
            market_data_port=_make_market_data_mock(), trade_diff_sink=InMemoryTradeDiffSink()
        )

    def test_send_order_returns_pending(self):
        result = self.broker.send_order("7203", OrderSide.BUY, 100)
        self.assertEqual(result["status"], "pending")
        self.assertIn("order_id", result)

    def test_send_order_saved_to_db(self):
        self.broker.send_order("7203", OrderSide.BUY, 100)
        with _db_connection() as con:
            row = con.execute("SELECT status FROM paper_orders WHERE symbol='7203'").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "pending")

    def test_cancel_order(self):
        result = self.broker.send_order("7203", OrderSide.BUY, 100)
        order_id = result["order_id"]
        cancel = self.broker.cancel_order(order_id)
        self.assertEqual(cancel["status"], "cancelled")
        with _db_connection() as con:
            row = con.execute(
                "SELECT status FROM paper_orders WHERE order_id=%s", [order_id]
            ).fetchone()
        self.assertEqual(row[0], "cancelled")

    def test_get_balance_initial(self):
        balance = self.broker.get_balance()
        self.assertEqual(balance, 1_000_000.0)

    def test_get_positions_empty(self):
        positions = self.broker.get_positions()
        self.assertEqual(positions, [])


class TestPaperBrokerSettle(unittest.TestCase):
    def setUp(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")
            con.execute("DELETE FROM paper_positions")
            con.execute("UPDATE paper_balance SET balance = 1000000.0")
        self._mock_market_data = _make_market_data_mock()
        self._trade_diff_sink = InMemoryTradeDiffSink()
        self.broker = PaperBroker(
            market_data_port=self._mock_market_data,
            trade_diff_sink=self._trade_diff_sink,
        )

    def test_settle_market_buy(self):
        self.broker.send_order("7203", OrderSide.BUY, 100)
        settled = self.broker.settle_pending_orders()
        self.assertEqual(len(settled), 1)
        self.assertAlmostEqual(settled[0]["fill_price"], 1000.0)
        # 残高が減少していること
        with _db_connection() as con:
            balance = con.execute("SELECT balance FROM paper_balance").fetchone()[0]
        self.assertAlmostEqual(balance, 1_000_000.0 - 1000.0 * 100)

    def test_settle_creates_position(self):
        self.broker.send_order("7203", OrderSide.BUY, 100)
        self.broker.settle_pending_orders()
        with _db_connection() as con:
            pos = con.execute(
                "SELECT qty, avg_price FROM paper_positions WHERE symbol='7203'"
            ).fetchone()
        self.assertIsNotNone(pos)
        self.assertEqual(pos[0], 100)

    def test_settle_sell_without_position_skips_cash(self):
        """未保有銘柄の SELL では現金・実現損益を計上しない（幻の現金防止）。"""
        self.broker.send_order("7203", OrderSide.SELL, 100)
        self.broker.settle_pending_orders()
        with _db_connection() as con:
            balance = con.execute("SELECT balance FROM paper_balance").fetchone()[0]
            pos = con.execute("SELECT qty FROM paper_positions WHERE symbol='7203'").fetchone()
            pnl = con.execute(
                "SELECT realized_pnl FROM paper_orders WHERE symbol='7203'"
            ).fetchone()[0]
        self.assertAlmostEqual(balance, 1_000_000.0)  # 増えていないこと
        self.assertIsNone(pos)
        self.assertIsNone(pnl)

    def test_settle_sell_clamps_to_held_qty(self):
        """保有数量を超える SELL は保有分にクランプして現金計上する。"""
        with _db_connection() as con:
            con.execute(
                "INSERT INTO paper_positions (symbol, qty, avg_price) VALUES ('7203', 100, 900.0)"
            )
        self.broker.send_order("7203", OrderSide.SELL, 150)  # 保有100を超える
        self.broker.settle_pending_orders()
        # fill=1000 のとき proceeds は 100株分のみ
        with _db_connection() as con:
            balance = con.execute("SELECT balance FROM paper_balance").fetchone()[0]
            pos = con.execute("SELECT qty FROM paper_positions WHERE symbol='7203'").fetchone()
            pnl = con.execute(
                "SELECT realized_pnl FROM paper_orders WHERE symbol='7203'"
            ).fetchone()[0]
        self.assertAlmostEqual(balance, 1_000_000.0 + 1000.0 * 100)
        self.assertIsNone(pos)  # 全量決済で削除
        self.assertAlmostEqual(pnl, (1000.0 - 900.0) * 100)

    def test_settle_skips_on_nan_ohlc(self):
        """NaN の OHLC では約定させず pending を据え置く（残高・avg_price 汚染防止）。"""
        nan_df = pd.DataFrame(
            {
                "Open": [float("nan")],
                "High": [float("nan")],
                "Low": [float("nan")],
                "Close": [1000.0],
            },
            index=pd.to_datetime(["2026-03-15"]),
        )
        self._mock_market_data.get_ohlcv.return_value = nan_df
        self.broker.send_order("7203", OrderSide.BUY, 100)
        settled = self.broker.settle_pending_orders()
        self.assertEqual(settled, [])
        with _db_connection() as con:
            status = con.execute("SELECT status FROM paper_orders WHERE symbol='7203'").fetchone()[
                0
            ]
            balance = con.execute("SELECT balance FROM paper_balance").fetchone()[0]
        self.assertEqual(status, "pending")
        self.assertAlmostEqual(balance, 1_000_000.0)

    def _set_predicted_at(self, order_id: str, predicted_at: str) -> None:
        with _db_connection() as con:
            con.execute(
                "UPDATE paper_orders SET market='jp', predicted_at=%s, "
                "signal_price=995.0 WHERE order_id=%s",
                [predicted_at, order_id],
            )

    def _assert_still_pending(self, order_id: str) -> None:
        with _db_connection() as con:
            status = con.execute(
                "SELECT status FROM paper_orders WHERE order_id=%s", [order_id]
            ).fetchone()[0]
            balance = con.execute("SELECT balance FROM paper_balance").fetchone()[0]
            pos = con.execute("SELECT qty FROM paper_positions WHERE symbol='7203'").fetchone()
        self.assertEqual(status, "pending")
        self.assertAlmostEqual(balance, 1_000_000.0)
        self.assertIsNone(pos)
        self.assertEqual(self._trade_diff_sink.recorded, [])

    def test_settle_defers_when_last_bar_is_signal_day(self):
        """最後の行がシグナルを作った日の行（当日の行がまだ無い）なら約定させず pending 据え置き。"""
        # 最後の行 = 2026-03-15、predicted_at = 2026-03-15 夜
        result = self.broker.send_order("7203", OrderSide.BUY, 100)
        self._set_predicted_at(result["order_id"], "20260315_200000")
        settled = self.broker.settle_pending_orders()
        self.assertEqual(settled, [])
        self._assert_still_pending(result["order_id"])

    def test_settle_defers_limit_order_without_expiring(self):
        """当日の行が無いときは指値未達でも失効させない（当日の値を見ていないため）。"""
        result = self.broker.send_order(
            "7203", OrderSide.BUY, 100, price=900.0, order_type=OrderType.LIMIT
        )
        self._set_predicted_at(result["order_id"], "20260316_200000")
        settled = self.broker.settle_pending_orders()
        self.assertEqual(settled, [])
        self._assert_still_pending(result["order_id"])

    def test_settle_compares_bar_date_in_jst(self):
        """tz つきの行の日付は JST に直して比べる（UTC では前日に見える当日の行で約定する）。"""
        jst_df = pd.DataFrame(
            {"Open": [1000.0], "High": [1050.0], "Low": [990.0], "Close": [1020.0]},
            index=pd.DatetimeIndex(["2026-03-16 00:00"]).tz_localize("Asia/Tokyo"),
        )
        self._mock_market_data.get_ohlcv.return_value = jst_df
        result = self.broker.send_order("7203", OrderSide.BUY, 100)
        self._set_predicted_at(result["order_id"], "20260315_200000")
        settled = self.broker.settle_pending_orders()
        self.assertEqual(len(settled), 1)
        self.assertAlmostEqual(settled[0]["fill_price"], 1000.0)

    def test_settle_defers_order_without_predicted_at_when_bar_is_not_today(self):
        """predicted_at を持たない注文（SL/TP・ルール・期限の手じまい）も、今日の行が無ければ据え置く（#805）"""
        # 最後の行は 03-15（前の営業日・休みの日の朝）
        with patch.object(paper_broker, "_today_jst", lambda: date(2026, 3, 16)):
            result = self.broker.send_order("7203", OrderSide.SELL, 100)
            settled = self.broker.settle_pending_orders()
        self.assertEqual(settled, [])
        with _db_connection() as con:
            status = con.execute(
                "SELECT status FROM paper_orders WHERE order_id=%s", [result["order_id"]]
            ).fetchone()[0]
        self.assertEqual(status, "pending")

    def test_settle_defers_when_bar_is_after_signal_but_before_today(self):
        """predicted_at より後でも、今日より前の行では約定しない（何日か据え置かれた注文）"""
        with patch.object(paper_broker, "_today_jst", lambda: date(2026, 3, 17)):
            result = self.broker.send_order("7203", OrderSide.BUY, 100)
            self._set_predicted_at(result["order_id"], "20260313_200000")
            settled = self.broker.settle_pending_orders()
        self.assertEqual(settled, [])
        self._assert_still_pending(result["order_id"])

    def test_settle_fills_order_without_predicted_at_on_today_bar(self):
        """predicted_at を持たない注文も、今日の行があれば今までどおり約定する"""
        self.broker.send_order("7203", OrderSide.BUY, 100)
        settled = self.broker.settle_pending_orders()
        self.assertEqual(len(settled), 1)

    def test_second_buy_while_pending_is_rejected(self):
        """同じ銘柄の買いが約定待ちのうちは、次の買いを受けない（重ね買いを防ぐ、#802）"""
        first = self.broker.send_order("7203", OrderSide.BUY, 100)
        second = self.broker.send_order("7203.T", OrderSide.BUY, 100)
        self.assertEqual(first["status"], "pending")
        self.assertEqual(second["status"], "rejected")
        self.assertEqual(second["order_id"], "")
        with _db_connection() as con:
            n = con.execute(
                "SELECT count(*) FROM paper_orders WHERE symbol='7203' AND status='pending'"
            ).fetchone()[0]
        self.assertEqual(n, 1)

    def test_buy_after_fill_is_accepted(self):
        self.broker.send_order("7203", OrderSide.BUY, 100)
        self.broker.settle_pending_orders()
        self.assertEqual(self.broker.send_order("7203", OrderSide.BUY, 100)["status"], "pending")

    def test_settle_rejects_buy_when_cash_is_short(self):
        """約定のときに現金が足りない買いは約定させず rejected にする（現金を負にしない、#802）"""
        with _db_connection() as con:
            con.execute("UPDATE paper_balance SET balance = 50000.0")
        result = self.broker.send_order("7203", OrderSide.BUY, 100)  # 100 * 1000 = 100000
        settled = self.broker.settle_pending_orders()
        self.assertEqual(settled, [])
        with _db_connection() as con:
            status = con.execute(
                "SELECT status FROM paper_orders WHERE order_id=%s", [result["order_id"]]
            ).fetchone()[0]
            balance = con.execute("SELECT balance FROM paper_balance").fetchone()[0]
            pos = con.execute("SELECT qty FROM paper_positions WHERE symbol='7203'").fetchone()
        self.assertEqual(status, "rejected")
        self.assertAlmostEqual(balance, 50000.0)
        self.assertIsNone(pos)

    def test_settle_fills_when_today_bar_present(self):
        """最後の行が predicted_at より後の営業日なら今までどおり約定する。"""
        result = self.broker.send_order("7203", OrderSide.BUY, 100)
        self._set_predicted_at(result["order_id"], "20260314_200000")
        settled = self.broker.settle_pending_orders()
        self.assertEqual(len(settled), 1)
        self.assertAlmostEqual(settled[0]["fill_price"], 1000.0)
        self.assertEqual(len(self._trade_diff_sink.recorded), 1)

    def test_settle_updates_paper_real_diff(self):
        result = self.broker.send_order("7203", OrderSide.BUY, 100)
        with _db_connection() as con:
            con.execute(
                "UPDATE paper_orders SET market='jp', predicted_at='20260314_085000', "
                "signal_price=995.0 WHERE order_id=%s",
                [result["order_id"]],
            )

        self.broker.settle_pending_orders()

        self.assertEqual(len(self._trade_diff_sink.recorded), 1)
        rec = self._trade_diff_sink.recorded[0]
        self.assertEqual(rec.market, "jp")
        self.assertEqual(rec.symbol, "7203")
        self.assertAlmostEqual(rec.actual_price, 1000.0)
        self.assertEqual(rec.mode, "paper")


class TestPaperBrokerShort(unittest.TestCase):
    def setUp(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")
            con.execute("DELETE FROM paper_positions")
            con.execute("DELETE FROM paper_short_positions")
            con.execute("UPDATE paper_balance SET balance = 1000000.0")
        self._mock_market_data = _make_market_data_mock()
        self.broker = PaperBroker(
            market_data_port=self._mock_market_data, trade_diff_sink=InMemoryTradeDiffSink()
        )

    def test_send_short_order_returns_pending(self):
        result = self.broker.send_order("7203", OrderSide.SHORT, 100, price=1500.0)
        self.assertEqual(result["status"], "pending")
        self.assertIn("order_id", result)

    def test_send_short_order_saved_with_side3(self):
        self.broker.send_order("7203", OrderSide.SHORT, 100, price=1500.0)
        with _db_connection() as con:
            row = con.execute("SELECT side FROM paper_orders WHERE symbol='7203'").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 3)

    def test_send_short_order_creates_short_position(self):
        """send_order(SHORT) で paper_short_positions に即時仮登録されること"""
        self.broker.send_order("7203", OrderSide.SHORT, 100, price=1500.0)
        with _db_connection() as con:
            pos = con.execute(
                "SELECT qty, avg_short_price FROM paper_short_positions WHERE symbol='7203'"
            ).fetchone()
        self.assertIsNotNone(pos)
        self.assertEqual(pos[0], 100)
        self.assertAlmostEqual(pos[1], 1500.0)

    def test_send_short_order_weighted_avg_update(self):
        """既存ポジションがあれば加重平均単価が更新されること"""
        with _db_connection() as con:
            con.execute(
                "INSERT INTO paper_short_positions"
                " (symbol, qty, avg_short_price) VALUES ('7203', 100, 1200.0)"
            )
        self.broker.send_order("7203", OrderSide.SHORT, 100, price=1400.0)
        with _db_connection() as con:
            pos = con.execute(
                "SELECT qty, avg_short_price FROM paper_short_positions WHERE symbol='7203'"
            ).fetchone()
        self.assertEqual(pos[0], 200)
        self.assertAlmostEqual(pos[1], 1300.0)  # (1200*100 + 1400*100) / 200

    def test_settle_short_updates_position_to_fill_price(self):
        """settle 後に paper_short_positions が実際の約定値段で更新されること"""
        # send_order で仮登録（price=1500）
        self.broker.send_order("7203", OrderSide.SHORT, 100, price=1500.0)
        settled = self.broker.settle_pending_orders()
        self.assertEqual(len(settled), 1)
        self.assertAlmostEqual(settled[0]["fill_price"], 1000.0)
        # settle 後は fill_price (open=1000) で上書きされること
        with _db_connection() as con:
            pos = con.execute(
                "SELECT qty, avg_short_price FROM paper_short_positions WHERE symbol='7203'"
            ).fetchone()
        self.assertIsNotNone(pos)
        self.assertEqual(pos[0], 100)
        self.assertAlmostEqual(pos[1], 1000.0)

    def test_settle_short_cover_computes_realized_pnl(self):
        """SHORT_COVER の約定で realized_pnl が正しく計算されること"""
        # 空売りポジション (avg=1200) を用意
        with _db_connection() as con:
            con.execute(
                "INSERT INTO paper_short_positions"
                " (symbol, qty, avg_short_price) VALUES ('7203', 100, 1200.0)"
            )
        result = self.broker.send_order("7203", OrderSide.SHORT_COVER, 100)
        self.broker.settle_pending_orders()
        with _db_connection() as con:
            row = con.execute(
                "SELECT realized_pnl FROM paper_orders WHERE order_id=%s",
                [result["order_id"]],
            ).fetchone()
        self.assertIsNotNone(row)
        # realized_pnl = (avg_short_price - fill_price) * qty = (1200 - 1000) * 100
        self.assertAlmostEqual(row[0], 20000.0)

    def test_settle_short_cover_reduces_position(self):
        """SHORT_COVER 約定後に paper_short_positions の qty が減少すること"""
        with _db_connection() as con:
            con.execute(
                "INSERT INTO paper_short_positions"
                " (symbol, qty, avg_short_price) VALUES ('7203', 200, 1200.0)"
            )
        self.broker.send_order("7203", OrderSide.SHORT_COVER, 100)
        self.broker.settle_pending_orders()
        with _db_connection() as con:
            pos = con.execute(
                "SELECT qty FROM paper_short_positions WHERE symbol='7203'"
            ).fetchone()
        self.assertIsNotNone(pos)
        self.assertEqual(pos[0], 100)

    def test_settle_short_cover_full_removes_position(self):
        """全数量返済で paper_short_positions レコードが削除されること"""
        with _db_connection() as con:
            con.execute(
                "INSERT INTO paper_short_positions"
                " (symbol, qty, avg_short_price) VALUES ('7203', 100, 1200.0)"
            )
        self.broker.send_order("7203", OrderSide.SHORT_COVER, 100)
        self.broker.settle_pending_orders()
        with _db_connection() as con:
            pos = con.execute(
                "SELECT qty FROM paper_short_positions WHERE symbol='7203'"
            ).fetchone()
        self.assertIsNone(pos)

    def test_get_short_positions(self):
        """get_short_positions が paper_short_positions を正しく返すこと"""
        with _db_connection() as con:
            con.execute(
                "INSERT INTO paper_short_positions"
                " (symbol, qty, avg_short_price) VALUES ('7203', 100, 1200.0)"
            )
        positions = self.broker.get_short_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["symbol"], "7203")
        self.assertEqual(positions[0]["qty"], 100)
        self.assertAlmostEqual(positions[0]["avg_short_price"], 1200.0)
        self.assertIn("unrealized_pnl", positions[0])

    def test_get_short_positions_empty(self):
        positions = self.broker.get_short_positions()
        self.assertEqual(positions, [])

    def _assert_short_unpriced(self):
        """現在値が取れない空売りは売値で埋めず NaN を返し、WARNING を出す（#844）"""
        with _db_connection() as con:
            con.execute(
                "INSERT INTO paper_short_positions"
                " (symbol, qty, avg_short_price) VALUES ('7203', 100, 1200.0)"
            )
        with self.assertLogs("src.trading.brokers.paper.paper_broker", level="WARNING"):
            positions = self.broker.get_short_positions()
        self.assertEqual(len(positions), 1)
        self.assertTrue(math.isnan(positions[0]["current_price"]))
        self.assertTrue(math.isnan(positions[0]["unrealized_pnl"]))

    def test_get_short_positions_unpriced_on_empty(self):
        self._mock_market_data.get_ohlcv.return_value = pd.DataFrame()
        self._assert_short_unpriced()

    def test_get_short_positions_unpriced_on_error(self):
        self._mock_market_data.get_ohlcv.side_effect = Exception("market data error")
        self._assert_short_unpriced()

    def test_get_short_positions_unpriced_on_nan_close(self):
        hist = _make_mock_ohlcv()
        hist["Close"] = float("nan")
        self._mock_market_data.get_ohlcv.return_value = hist
        self._assert_short_unpriced()


class TestPaperBrokerGetToken(unittest.TestCase):
    def setUp(self):
        self.broker = PaperBroker(
            market_data_port=_make_market_data_mock(), trade_diff_sink=InMemoryTradeDiffSink()
        )

    def test_get_token_returns_paper_mode(self):
        self.assertEqual(self.broker.get_token(), "paper_mode")

    def test_get_token_returns_string(self):
        self.assertIsInstance(self.broker.get_token(), str)


class TestPaperBrokerGetBalance(unittest.TestCase):
    def setUp(self):
        with _db_connection() as con:
            con.execute("UPDATE paper_balance SET balance = 1000000.0")
        self.broker = PaperBroker(
            market_data_port=_make_market_data_mock(), trade_diff_sink=InMemoryTradeDiffSink()
        )

    def test_get_balance_returns_current_value(self):
        with _db_connection() as con:
            con.execute("UPDATE paper_balance SET balance = 500000.0")
        balance = self.broker.get_balance()
        self.assertAlmostEqual(balance, 500000.0)

    def test_get_balance_returns_initial_when_table_empty(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_balance")
        balance = self.broker.get_balance()
        from config.settings import PAPER_INITIAL_BALANCE

        self.assertAlmostEqual(balance, PAPER_INITIAL_BALANCE)


class TestPaperBrokerGetOrders(unittest.TestCase):
    def setUp(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")
            con.execute("DELETE FROM paper_positions")
            con.execute("UPDATE paper_balance SET balance = 1000000.0")
        self.broker = PaperBroker(
            market_data_port=_make_market_data_mock(), trade_diff_sink=InMemoryTradeDiffSink()
        )

    def test_get_orders_returns_list(self):
        result = self.broker.get_orders()
        self.assertIsInstance(result, list)

    def test_get_orders_empty_when_no_orders(self):
        result = self.broker.get_orders()
        self.assertEqual(result, [])

    def test_get_orders_contains_todays_order(self):
        self.broker.send_order("7203", OrderSide.BUY, 100, price=1000.0)
        orders = self.broker.get_orders()
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["symbol"], "7203")
        self.assertEqual(orders[0]["side"], int(OrderSide.BUY))
        self.assertEqual(orders[0]["status"], "pending")

    def test_get_orders_includes_required_keys(self):
        self.broker.send_order("9984", OrderSide.SELL, 50, price=2000.0)
        orders = self.broker.get_orders()
        self.assertEqual(len(orders), 1)
        for key in ("order_id", "symbol", "side", "qty", "price", "status"):
            self.assertIn(key, orders[0])
        self.assertEqual(orders[0]["qty"], 50)


class TestPaperBrokerGetPositionsAdditional(unittest.TestCase):
    def setUp(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")
            con.execute("DELETE FROM paper_positions")
            con.execute("UPDATE paper_balance SET balance = 1000000.0")
        self._mock_market_data = _make_market_data_mock()
        self.broker = PaperBroker(
            market_data_port=self._mock_market_data, trade_diff_sink=InMemoryTradeDiffSink()
        )

    def test_get_positions_returns_position_with_pnl(self):
        with _db_connection() as con:
            con.execute(
                "INSERT INTO paper_positions (symbol, qty, avg_price) VALUES ('7203', 100, 1000.0)"
            )
        positions = self.broker.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["symbol"], "7203")
        self.assertEqual(positions[0]["qty"], 100)
        self.assertAlmostEqual(positions[0]["avg_price"], 1000.0)
        self.assertIn("unrealized_pnl", positions[0])
        self.assertAlmostEqual(positions[0]["unrealized_pnl"], 2000.0)

    def _assert_unpriced(self):
        """現在値が取れない保有は買値で埋めず NaN を返し、WARNING を出す（#844）"""
        with _db_connection() as con:
            con.execute(
                "INSERT INTO paper_positions (symbol, qty, avg_price) VALUES ('7203', 100, 1000.0)"
            )
        with self.assertLogs("src.trading.brokers.paper.paper_broker", level="WARNING"):
            positions = self.broker.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertAlmostEqual(positions[0]["avg_price"], 1000.0)
        self.assertTrue(math.isnan(positions[0]["current_price"]))
        self.assertTrue(math.isnan(positions[0]["unrealized_pnl"]))

    def test_get_positions_unpriced_on_yf_error(self):
        self._mock_market_data.get_ohlcv.side_effect = Exception("market data error")
        self._assert_unpriced()

    def test_get_positions_unpriced_on_empty(self):
        self._mock_market_data.get_ohlcv.return_value = pd.DataFrame()
        self._assert_unpriced()

    def test_get_positions_unpriced_on_nan_close(self):
        hist = _make_mock_ohlcv()
        hist["Close"] = float("nan")
        self._mock_market_data.get_ohlcv.return_value = hist
        self._assert_unpriced()

    def test_get_positions_unpriced_on_stale_bar(self):
        """最後の足が何日も前（売買停止など）なら、その終値を今の値として使わない"""
        self._mock_market_data.get_ohlcv.return_value = pd.DataFrame(
            {"Open": [1000.0], "High": [1050.0], "Low": [990.0], "Close": [1020.0]},
            index=pd.to_datetime(["2026-03-01"]),
        )
        self._assert_unpriced()

    def test_get_positions_uses_previous_close_over_weekend(self):
        """休みの日の前の営業日の足は、今の値として使う（取れたときの今の動き）"""
        self._mock_market_data.get_ohlcv.return_value = pd.DataFrame(
            {"Open": [1000.0], "High": [1050.0], "Low": [990.0], "Close": [1020.0]},
            index=pd.to_datetime(["2026-03-13"]),
        )
        with _db_connection() as con:
            con.execute(
                "INSERT INTO paper_positions (symbol, qty, avg_price) VALUES ('7203', 100, 1000.0)"
            )
        positions = self.broker.get_positions()
        self.assertAlmostEqual(positions[0]["current_price"], 1020.0)
        self.assertAlmostEqual(positions[0]["unrealized_pnl"], 2000.0)


class TestPaperBrokerGetPnlSummary(unittest.TestCase):
    def setUp(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")
            con.execute("DELETE FROM paper_positions")
            con.execute("UPDATE paper_balance SET balance = 1000000.0")
        self.broker = PaperBroker(
            market_data_port=_make_market_data_mock(), trade_diff_sink=InMemoryTradeDiffSink()
        )

    def test_get_pnl_summary_returns_dict(self):
        result = self.broker.get_pnl_summary()
        self.assertIsInstance(result, dict)

    def test_get_pnl_summary_zero_when_no_trades(self):
        result = self.broker.get_pnl_summary()
        self.assertAlmostEqual(result["realized_pnl"], 0.0)
        self.assertEqual(result["trade_count"], 0)
        self.assertIsNone(result["started_at"])

    def test_get_pnl_summary_contains_balance(self):
        result = self.broker.get_pnl_summary()
        self.assertAlmostEqual(result["balance"], 1000000.0)
        from config.settings import PAPER_INITIAL_BALANCE

        self.assertAlmostEqual(result["initial_balance"], PAPER_INITIAL_BALANCE)

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

    def test_get_pnl_summary_with_filled_sell_order(self):
        with _db_connection() as con:
            con.execute("""
                INSERT INTO paper_orders
                    (order_id, symbol, side, qty, price, order_type,
                     status, realized_pnl, filled_at)
                VALUES ('test_pnl_01', '7203', 2, 100, 1000.0, 10,
                        'filled', 5000.0, CURRENT_TIMESTAMP)
                """)
        result = self.broker.get_pnl_summary()
        self.assertAlmostEqual(result["realized_pnl"], 5000.0)
        self.assertEqual(result["trade_count"], 1)
        self.assertIsNotNone(result["started_at"])

    def test_get_pnl_summary_total_pnl_equals_realized_plus_unrealized(self):
        result = self.broker.get_pnl_summary()
        self.assertAlmostEqual(
            result["total_pnl"],
            result["realized_pnl"] + result["unrealized_pnl"],
        )


if __name__ == "__main__":
    unittest.main()
