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
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

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

    def _insert_order(
        self,
        order_id: str,
        symbol: str,
        side: OrderSide,
        status: str,
        target_exit_date: str | None = None,
        filled_at: str | None = None,
    ) -> None:
        with _db_connection() as con:
            con.execute(
                "INSERT INTO paper_orders "
                "(order_id, symbol, side, qty, order_type, status, target_exit_date, filled_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                [order_id, symbol, int(side), 100, 10, status, target_exit_date, filled_at],
            )

    @patch(_BROKER_PATCH)
    def test_skip_if_latest_buy_not_yet_due(self, mock_broker_cls):
        """期限切れで売り終えた古い BUY があっても、今の保有を作った最新の
        filled BUY が期限前なら売らない (Issue #945)。"""
        from src.orchestration.scheduler import run_horizon_exit_check

        today = date.today()
        self._insert_order(
            "ORD-200",
            "8766",
            OrderSide.BUY,
            "filled",
            target_exit_date=(today - timedelta(days=10)).isoformat(),
            filled_at=f"{(today - timedelta(days=12)).isoformat()} 00:05:00",
        )
        self._insert_order(
            "ORD-201",
            "8766",
            OrderSide.SELL,
            "filled",
            filled_at=f"{(today - timedelta(days=9)).isoformat()} 00:05:00",
        )
        self._insert_order(
            "ORD-202",
            "8766",
            OrderSide.BUY,
            "filled",
            target_exit_date=(today + timedelta(days=3)).isoformat(),
            filled_at=f"{(today - timedelta(days=1)).isoformat()} 00:05:00",
        )

        broker = MagicMock()
        broker.get_positions.return_value = [{"symbol": "8766", "qty": 100}]
        mock_broker_cls.return_value = broker

        run_horizon_exit_check()

        broker.send_order.assert_not_called()

    @patch(_BROKER_PATCH)
    def test_skip_if_pending_sell_exists(self, mock_broker_cls):
        """同じ銘柄に約定前の SELL があるときは売りを重ねない (Issue #945)。"""
        from src.orchestration.scheduler import run_horizon_exit_check

        self._insert_overdue_order("ORD-210", "8766")
        self._insert_order("ORD-211", "8766", OrderSide.SELL, "pending")

        broker = MagicMock()
        broker.get_positions.return_value = [{"symbol": "8766", "qty": 100}]
        mock_broker_cls.return_value = broker

        run_horizon_exit_check()

        broker.send_order.assert_not_called()

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
    def test_reads_positions_once_for_all_overdue(self, mock_broker_cls):
        """期限切れが複数でも保有の一覧はループの前に1回だけ読み、写しに無い・株数0は飛ばす (#1095)。"""
        from src.orchestration.scheduler import run_horizon_exit_check

        for i, symbol in enumerate(("7203", "8766", "9984", "6758")):
            self._insert_overdue_order(f"ORD-12{i}", symbol)

        broker = MagicMock()
        broker.get_positions.return_value = [
            {"symbol": "7203", "qty": 100},
            {"symbol": "8766.T", "qty": 300},
            {"symbol": "9984", "qty": 0},
        ]
        mock_broker_cls.return_value = broker

        run_horizon_exit_check()

        self.assertEqual(broker.get_positions.call_count, 1)
        self.assertEqual(broker.send_order.call_count, 2)
        broker.send_order.assert_any_call("7203", OrderSide.SELL, 100)
        broker.send_order.assert_any_call("8766", OrderSide.SELL, 300)

    @patch(_BROKER_PATCH)
    def test_one_symbol_failure_exits_others_then_fails_job(self, mock_broker_cls):
        """1銘柄の決済が落ちても他の銘柄は決済し、最後にジョブを失敗にする (#1074)。"""
        from src.orchestration.scheduler import run_horizon_exit_check

        self._insert_overdue_order("ORD-110", "7203")
        self._insert_overdue_order("ORD-111", "9984")

        broker = MagicMock()
        broker.get_positions.return_value = [
            {"symbol": "7203", "qty": 100},
            {"symbol": "9984", "qty": 200},
        ]
        broker.send_order.side_effect = [RuntimeError("broker down"), MagicMock()]
        mock_broker_cls.return_value = broker

        with self.assertLogs("src.orchestration.jobs.daily", level="ERROR"):
            with self.assertRaises(RuntimeError):
                run_horizon_exit_check()

        self.assertEqual(broker.send_order.call_count, 2)
        broker.send_order.assert_any_call("9984", OrderSide.SELL, 200)

    @patch(_BROKER_PATCH)
    def test_covers_overdue_short_positions_only(self, mock_broker_cls):
        """最後の filled の SHORT の期限が過ぎた空売りだけを SHORT_COVER の成行で返済し、
        期限内の空売り・買い持ちには出さない。side ごとに最後の約定を見る (#1145)。"""
        from src.orchestration.scheduler import run_horizon_exit_check

        today = date.today()
        overdue = (today - timedelta(days=1)).isoformat()
        not_due = (today + timedelta(days=3)).isoformat()
        filled = f"{(today - timedelta(days=2)).isoformat()} 00:05:00"
        later = f"{(today - timedelta(days=1)).isoformat()} 00:05:00"
        self._insert_order("ORD-300", "7203", OrderSide.SHORT, "filled", overdue, filled)
        self._insert_order("ORD-301", "9984", OrderSide.SHORT, "filled", not_due, filled)
        self._insert_order("ORD-302", "6758", OrderSide.BUY, "filled", not_due, filled)
        # 同じ銘柄の後の SHORT の期限が過ぎていても、買い持ちの期限は BUY の行で決める
        self._insert_order("ORD-303", "8766", OrderSide.BUY, "filled", not_due, filled)
        self._insert_order("ORD-304", "8766", OrderSide.SHORT, "filled", overdue, later)

        broker = MagicMock()
        broker.get_positions.return_value = [
            {"symbol": "6758", "qty": 100},
            {"symbol": "8766", "qty": 100},
        ]
        broker.get_short_positions.return_value = [
            {"symbol": "7203.T", "qty": 300},
            {"symbol": "9984", "qty": 200},
            {"symbol": "8766", "qty": 400},
        ]
        mock_broker_cls.return_value = broker

        run_horizon_exit_check()

        self.assertEqual(broker.send_order.call_count, 2)
        broker.send_order.assert_any_call("7203", OrderSide.SHORT_COVER, 300)
        broker.send_order.assert_any_call("8766", OrderSide.SHORT_COVER, 400)

    @patch(_BROKER_PATCH)
    def test_short_positions_unsupported_treated_as_empty(self, mock_broker_cls):
        """get_short_positions が無い broker でも買い持ちの決済は続ける (#1145)。"""
        from src.orchestration.scheduler import run_horizon_exit_check

        self._insert_overdue_order("ORD-310", "7203")
        self._insert_order(
            "ORD-311",
            "9984",
            OrderSide.SHORT,
            "filled",
            (date.today() - timedelta(days=1)).isoformat(),
        )

        broker = MagicMock()
        broker.get_positions.return_value = [{"symbol": "7203", "qty": 100}]
        broker.get_short_positions.side_effect = AttributeError
        mock_broker_cls.return_value = broker

        run_horizon_exit_check()

        broker.send_order.assert_called_once_with("7203", OrderSide.SELL, 100)

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


_JST = ZoneInfo("Asia/Tokyo")
# 2026-10-24 JST の定時の回（08:50 JST＝前の日の 23:50 UTC）と補完の回（09:05 JST＝00:05 UTC）
_SCHEDULED_UTC = datetime(2026, 10, 23, 23, 50, tzinfo=timezone.utc)
_RECOVERY_UTC = datetime(2026, 10, 24, 0, 5, tzinfo=timezone.utc)


@contextmanager
def _clock(utc_now: datetime) -> Iterator[None]:
    """UTC のコンテナの時計を utc_now に置く（date.today() は UTC の日付を返す）。"""
    fake_date = type("_FakeDate", (date,), {"today": classmethod(lambda cls: utc_now.date())})
    with (
        patch("datetime.date", fake_date),
        patch("src.trading.execution.recording.date", fake_date, create=True),
        patch(
            "src.trading.execution.recording.now_jst",
            lambda: utc_now.astimezone(_JST),
            create=True,
        ),
        patch("src.orchestration.jobs.common._local_now", lambda: utc_now),
    ):
        yield


class TestHorizonExitJstDate(unittest.TestCase):
    """期限と決済の「今日」を JST の日付で決め、回す時刻で売る日が変わらない (Issue #949)。"""

    def setUp(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")

    def _buy_at(self, utc_now: datetime, order_id: str, symbol: str, horizon: int) -> None:
        from src.infrastructure.in_memory import InMemoryTradeDiffSink
        from src.trading.brokers.base import OrderType
        from src.trading.execution.recording import _record_order

        with _db_connection() as con:
            con.execute(
                "INSERT INTO paper_orders "
                "(order_id, symbol, side, qty, order_type, status, filled_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [order_id, symbol, int(OrderSide.BUY), 100, 10, "filled", utc_now],
            )
        with _clock(utc_now):
            _record_order(
                market="jp",
                predicted_at="20261023_060000",
                symbol=symbol,
                side=OrderSide.BUY,
                qty=100,
                signal_price=1000.0,
                order_price=1000.0,
                order_type=OrderType.MARKET,
                order_result={"order_id": order_id, "fill_price": 1000.0},
                broker=None,
                mode="paper",
                horizon=horizon,
                trade_diff_sink=InMemoryTradeDiffSink(),
            )

    def _sold_at(self, utc_now: datetime, symbol: str) -> bool:
        from src.orchestration.scheduler import run_horizon_exit_check

        with patch(_BROKER_PATCH) as mock_broker_cls:
            broker = MagicMock()
            broker.get_positions.return_value = [{"symbol": symbol, "qty": 100}]
            mock_broker_cls.return_value = broker
            with _clock(utc_now):
                run_horizon_exit_check()
        return broker.send_order.called

    def _first_sell_day(self, symbol: str) -> date:
        """発注の後の各 JST の日の定時の回（08:50 JST）で、初めて売る日を返す。"""
        for days in range(1, 8):
            run_utc = _SCHEDULED_UTC + timedelta(days=days)
            if self._sold_at(run_utc, symbol):
                return run_utc.astimezone(_JST).date()
        self.fail(f"{symbol} を売らなかった")

    def test_sell_day_same_for_scheduled_and_recovery_runs(self):
        self._buy_at(_SCHEDULED_UTC, "ORD-949-S", "7203", horizon=3)
        scheduled_day = self._first_sell_day("7203")
        self.setUp()
        self._buy_at(_RECOVERY_UTC, "ORD-949-R", "7203", horizon=3)
        recovery_day = self._first_sell_day("7203")

        self.assertEqual(scheduled_day, recovery_day)
        # 2026-10-24（土）を1日目に 10-26（月）・10-27（火）と営業日で数え、10-28 の朝に売る (#1119)
        self.assertEqual(scheduled_day, date(2026, 10, 28))

    def test_recovery_run_does_not_exit_position_before_due(self):
        # 2026-10-23（金）JST の定時の回で出した horizon=2 の BUY は、期限が 10-26（月）で
        # 2026-10-27 の朝に売る (#1119)。2026-10-24 JST の補完の回と 10-25 の回では売らない。
        self._buy_at(_SCHEDULED_UTC - timedelta(days=1), "ORD-949-E", "9984", horizon=2)

        self.assertFalse(self._sold_at(_RECOVERY_UTC, "9984"))
        self.assertFalse(self._sold_at(_SCHEDULED_UTC + timedelta(days=1), "9984"))
        self.assertTrue(self._sold_at(_SCHEDULED_UTC + timedelta(days=3), "9984"))


if __name__ == "__main__":
    unittest.main()
