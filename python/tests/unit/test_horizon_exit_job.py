"""ホライズン決済の独立のジョブ（#1321）

8:50 の自動発注のジョブがデプロイの起き直しで決済の前に消えても、同じ朝のうちに
期限を過ぎた保有を決済し直す。8:50 の回が出した約定待ちの SELL には重ねない。
"""

import json
import os
import unittest
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

from src.orchestration.scheduler_queue import SchedulerQueueManager
from src.trading.brokers.base import OrderSide
from src.utils.db._connection import _db_connection

_JOB_ID = "daily_horizon_exit"
_BROKER_PATCH = "src.trading.brokers.paper.paper_broker.PaperBroker"
_DB_PATCH = "src.utils.db._connection._db_connection"


def _write_started(state_path, job_id: str, period_key: str) -> None:
    """前の起動のプロセスが始めの記録だけを残して消えた状態を作る"""
    event = {
        "job_id": job_id,
        "reason": "scheduled",
        "status": "started",
        "period_key": period_key,
        "run_id": "run-a",
        "boot_id": "previous-boot",
        "started_at": "2020-01-01T00:00:00+00:00",
    }
    state_path.write_text(json.dumps({"events": [event]}), encoding="utf-8")


class TestHorizonExitJobConfig(unittest.TestCase):
    def test_job_is_scheduled_after_auto_order_and_reruns_if_interrupted(self):
        """8:50 の発注の後・9:05 の約定の前に走り、消えた回を 9:30 まで補完する"""
        from run_scheduler import SCHEDULE_CONFIG

        config = SCHEDULE_CONFIG[_JOB_ID]
        auto_order = SCHEDULE_CONFIG["daily_auto_order"]
        settle = SCHEDULE_CONFIG["daily_settle_orders"]
        self.assertTrue(config["rerun_if_interrupted"])
        self.assertEqual(config["recovery_until"], (9, 30))
        self.assertEqual(config["recovery_until"], auto_order["recovery_until"])
        self.assertEqual(config["day_of_week"], auto_order["day_of_week"])
        when = (config["hour"], config["minute"])
        self.assertGreater(when, (auto_order["hour"], auto_order["minute"]))
        self.assertLess(when, (settle["hour"], settle["minute"]))
        # 発注のジョブなので、同じ回の中で頭から呼び直さない（#836）
        self.assertEqual(config["max_retries"], 0)
        self.assertEqual(config["max_executions_per_period"], 1)
        self.assertTrue(config.get("auto_schedule", True))

    def test_auto_order_job_is_still_not_rerun(self):
        """8:50 の発注のジョブ自体は今までどおり補完しない（#946）"""
        from run_scheduler import SCHEDULE_CONFIG

        self.assertFalse(SCHEDULE_CONFIG["daily_auto_order"].get("rerun_if_interrupted", False))

    def test_job_calls_horizon_exit_check(self):
        import run_scheduler

        with patch("src.orchestration.scheduler.run_horizon_exit_check") as check:
            run_scheduler.SCHEDULE_CONFIG[_JOB_ID]["func"]()

        check.assert_called_once_with()


def test_interrupted_run_is_recovered_until_cutoff(tmp_path):
    """前の起動で消えた回を、同じ朝の 9:30 の締めまで補完してよいと返す"""
    from run_scheduler import SCHEDULE_CONFIG

    state_path = tmp_path / "state.json"
    # 2026-10-28 は水曜
    _write_started(state_path, _JOB_ID, "2026-10-28")
    manager = SchedulerQueueManager(SCHEDULE_CONFIG, state_file_path=str(state_path))

    before_cutoff = datetime(2026, 10, 28, 9, 20, tzinfo=manager.tz)
    assert manager.should_recover(_JOB_ID, now=before_cutoff) is True
    at_cutoff = datetime(2026, 10, 28, 9, 30, tzinfo=manager.tz)
    assert manager.should_recover(_JOB_ID, now=at_cutoff) is False


class TestHorizonExitJobOrders(unittest.TestCase):
    """足したジョブが 8:50 の回の決済に重ねず、漏れた決済だけを出す"""

    def setUp(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")

    def _insert(self, order_id: str, side: OrderSide, status: str, exit_date=None) -> None:
        with _db_connection() as con:
            con.execute(
                "INSERT INTO paper_orders "
                "(order_id, symbol, side, qty, order_type, status, target_exit_date) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [order_id, "8766", int(side), 100, 10, status, exit_date],
            )

    def _run_job(self) -> MagicMock:
        from run_scheduler import SCHEDULE_CONFIG

        with patch(_BROKER_PATCH) as mock_broker_cls:
            broker = MagicMock()
            broker.get_positions.return_value = [{"symbol": "8766", "qty": 100}]
            mock_broker_cls.return_value = broker
            SCHEDULE_CONFIG[_JOB_ID]["func"]()
        return broker

    def test_no_second_sell_when_auto_order_already_sent_one(self):
        """8:50 の回のホライズン決済の SELL が約定待ちなら、2本目の SELL を出さない"""
        overdue = (date.today() - timedelta(days=1)).isoformat()
        self._insert("ORD-1", OrderSide.BUY, "filled", overdue)
        self._insert("ORD-2", OrderSide.SELL, "pending")

        broker = self._run_job()

        broker.send_order.assert_not_called()

    def test_sells_overdue_position_when_auto_order_died_before_exit(self):
        """8:50 の回が決済の前に消えた（約定待ちの SELL が無い）なら、期限を過ぎた保有を売る"""
        overdue = (date.today() - timedelta(days=1)).isoformat()
        self._insert("ORD-1", OrderSide.BUY, "filled", overdue)

        broker = self._run_job()

        broker.send_order.assert_called_once_with("8766", OrderSide.SELL, 100)

    @patch(_BROKER_PATCH)
    @patch(_DB_PATCH)
    def test_live_mode_does_nothing(self, mock_db, mock_broker_cls):
        from run_scheduler import SCHEDULE_CONFIG

        with patch.dict(os.environ, {"AUTO_TRADE_MODE": "live"}):
            SCHEDULE_CONFIG[_JOB_ID]["func"]()

        mock_db.assert_not_called()
        mock_broker_cls.assert_not_called()


class TestHorizonExitCheckIsSerialized(unittest.TestCase):
    def test_concurrent_calls_do_not_overlap(self):
        """8:50 の回と 9:00 のジョブが同時に呼んでも、決済の判定は1回ずつ通る（#1321）"""
        import threading
        import time

        from src.orchestration.jobs.daily import run_horizon_exit_check

        active = 0
        max_active = 0
        guard = threading.Lock()

        def _slow_exit():
            nonlocal active, max_active
            with guard:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with guard:
                active -= 1
            return []

        with (
            patch.dict(os.environ, {"AUTO_TRADE_MODE": "paper"}),
            patch(
                "src.orchestration.jobs.horizon_exit.exit_expired_positions",
                side_effect=_slow_exit,
            ) as mock_exit,
        ):
            threads = [threading.Thread(target=run_horizon_exit_check) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.assertEqual(mock_exit.call_count, 2)
        self.assertEqual(max_active, 1)
