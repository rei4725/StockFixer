"""約定のやり直しのジョブのテスト（#1008）"""

import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

_JST = ZoneInfo("Asia/Tokyo")


class TestRunDailySettleRetry(unittest.TestCase):
    """run_daily_settle_retry のテスト"""

    def _run(self, pending_counts, now):
        from src.orchestration.jobs import settle_retry

        with (
            patch.object(
                settle_retry, "count_pending_paper_orders", side_effect=pending_counts
            ) as mock_count,
            patch.object(settle_retry, "run_daily_settle_orders") as mock_settle,
            patch("src.reporting.discord.discord_utils.send_webhook_notification") as mock_notify,
            patch.object(settle_retry.logger, "warning") as mock_warning,
        ):
            settle_retry.run_daily_settle_retry(now=now)
        return mock_settle, mock_count, mock_notify, mock_warning

    def test_deferred_order_is_settled_in_same_day_retry(self):
        """9:05 に据え置いた注文を、同じ日のやり直しの回で約定の処理に回す"""
        now = datetime(2026, 10, 22, 9, 15, tzinfo=_JST)
        settle, _, notify, warning = self._run([1], now)
        settle.assert_called_once_with()
        notify.assert_not_called()
        warning.assert_not_called()

    def test_no_pending_does_not_run_settle_again(self):
        """据え置きが無い日は、約定の処理をもう一度動かさない"""
        for minute in (15, 25, 35, 45, 55):
            now = datetime(2026, 10, 22, 9, minute, tzinfo=_JST)
            settle, count, notify, _ = self._run([0], now)
            settle.assert_not_called()
            count.assert_called_once_with()
            notify.assert_not_called()

    def test_remaining_before_cutoff_does_not_warn(self):
        """締めの前の回で残っても、知らせを出さない"""
        now = datetime(2026, 10, 22, 9, 45, tzinfo=_JST)
        settle, _, notify, warning = self._run([1, 1], now)
        settle.assert_called_once_with()
        notify.assert_not_called()
        warning.assert_not_called()

    def test_remaining_at_cutoff_warns_once(self):
        """締めの回で残った注文は据え置いたまま、WARNING と知らせを1回だけ出す"""
        now = datetime(2026, 10, 22, 9, 55, tzinfo=_JST)
        settle, count, notify, warning = self._run([2, 1], now)
        settle.assert_called_once_with()
        self.assertEqual(count.call_count, 2)
        notify.assert_called_once()
        warning.assert_called_once()
        self.assertIn("1 件", notify.call_args.args[1])

    def test_all_settled_at_cutoff_does_not_warn(self):
        """締めの回で全部約定したら、知らせを出さない"""
        now = datetime(2026, 10, 22, 9, 55, tzinfo=_JST)
        settle, _, notify, warning = self._run([1, 0], now)
        settle.assert_called_once_with()
        notify.assert_not_called()
        warning.assert_not_called()

    def test_notify_failure_is_logged(self):
        """知らせが失敗しても、ジョブは例外を出さずに終える"""
        from src.orchestration.jobs import settle_retry

        now = datetime(2026, 10, 22, 9, 55, tzinfo=_JST)
        with (
            patch.object(settle_retry, "count_pending_paper_orders", side_effect=[1, 1]),
            patch.object(settle_retry, "run_daily_settle_orders"),
            patch(
                "src.reporting.discord.discord_utils.send_webhook_notification",
                side_effect=RuntimeError("down"),
            ),
            patch.object(settle_retry.logger, "error") as mock_error,
        ):
            settle_retry.run_daily_settle_retry(now=now)
        mock_error.assert_called_once()


class TestCountPendingPaperOrders(unittest.TestCase):
    """count_pending_paper_orders のテスト"""

    def test_counts_only_pending(self):
        from src.orchestration.jobs import settle_retry

        with patch("src.utils.db._connection._db_connection") as mock_conn:
            con = mock_conn.return_value.__enter__.return_value
            con.execute.return_value.fetchone.return_value = (3,)
            self.assertEqual(settle_retry.count_pending_paper_orders(), 3)
        sql = con.execute.call_args.args[0]
        self.assertIn("status = 'pending'", sql)


class TestSettleRetrySchedule(unittest.TestCase):
    """約定のやり直しの回の設定"""

    def test_retry_runs_until_cutoff_and_main_settle_stays_once(self):
        import run_scheduler

        cfg = run_scheduler.SCHEDULE_CONFIG
        self.assertEqual(cfg["daily_settle_orders"]["max_executions_per_period"], 1)
        retry = cfg["daily_settle_orders_retry"]
        self.assertEqual(retry["hour"], 9)
        self.assertEqual(retry["minute"], "15,25,35,45,55")
        self.assertEqual(retry["max_executions_per_period"], 5)
        self.assertEqual(retry["max_retries"], 0)
        self.assertNotIn("recovery_until", retry)

    def test_retry_job_calls_settle_retry(self):
        import run_scheduler

        with patch("src.orchestration.jobs.settle_retry.run_daily_settle_retry") as mock_retry:
            run_scheduler.job_daily_settle_orders_retry()
        mock_retry.assert_called_once_with()
