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
            patch.object(settle_retry, "run_daily_settle_orders", return_value=0) as mock_settle,
            patch("src.reporting.discord.discord_utils.send_webhook_notification") as mock_notify,
            patch.object(settle_retry.logger, "warning") as mock_warning,
        ):
            settle_retry.run_daily_settle_retry(now=now)
        return mock_settle, mock_count, mock_notify, mock_warning

    def test_deferred_order_is_settled_in_same_day_retry(self):
        """9:05 に据え置いた注文を、同じ日のやり直しの回で約定の処理に回す"""
        now = datetime(2026, 10, 22, 9, 15, tzinfo=_JST)
        settle, _, notify, warning = self._run([1], now)
        settle.assert_called_once_with(notify_when_none=False)
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
        settle.assert_called_once_with(notify_when_none=False)
        notify.assert_not_called()
        warning.assert_not_called()

    def test_remaining_at_cutoff_warns_once(self):
        """締めの回で残った注文は据え置いたまま、WARNING と知らせを1回だけ出す"""
        now = datetime(2026, 10, 22, 9, 55, tzinfo=_JST)
        settle, count, notify, warning = self._run([2, 1], now)
        settle.assert_called_once_with(notify_when_none=False)
        self.assertEqual(count.call_count, 2)
        notify.assert_called_once()
        warning.assert_called_once()
        self.assertIn("1 件", notify.call_args.args[1])

    def test_all_settled_at_cutoff_does_not_warn(self):
        """締めの回で全部約定したら、知らせを出さない"""
        now = datetime(2026, 10, 22, 9, 55, tzinfo=_JST)
        settle, _, notify, warning = self._run([1, 0], now)
        settle.assert_called_once_with(notify_when_none=False)
        notify.assert_not_called()
        warning.assert_not_called()

    def test_notify_failure_is_logged(self):
        """知らせが失敗しても、ジョブは例外を出さずに終える"""
        from src.orchestration.jobs import settle_retry

        now = datetime(2026, 10, 22, 9, 55, tzinfo=_JST)
        with (
            patch.object(settle_retry, "count_pending_paper_orders", side_effect=[1, 1]),
            patch.object(settle_retry, "run_daily_settle_orders", return_value=0),
            patch(
                "src.reporting.discord.discord_utils.send_webhook_notification",
                side_effect=RuntimeError("down"),
            ),
            patch.object(settle_retry.logger, "error") as mock_error,
        ):
            settle_retry.run_daily_settle_retry(now=now)
        mock_error.assert_called_once()


class TestSettleCompletionNotifyOnce(unittest.TestCase):
    """約定待ちがある日の完了の知らせの回数（#1047）"""

    def _run_morning(self, settled_per_run, pending_per_retry):
        """9:05 の約定と、やり直しの5回を順に流し、完了の知らせの件数の列を返す"""
        from src.orchestration.jobs import daily, settle_retry

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("src.trading.brokers.paper.paper_broker.PaperBroker") as mock_broker_cls,
            patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter"),
            patch("src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink"),
            patch("src.reporting.discord.discord_utils.send_daily_settle_completion") as mock_send,
            patch.object(settle_retry, "count_pending_paper_orders", side_effect=pending_per_retry),
        ):
            mock_broker_cls.return_value.settle_pending_orders.side_effect = settled_per_run
            mock_broker_cls.return_value.last_settle_errors = 0
            daily.run_daily_settle_orders()
            for minute in (15, 25, 35, 45, 55):
                now = datetime(2026, 10, 22, 9, minute, tzinfo=_JST)
                settle_retry.run_daily_settle_retry(now=now)
        return [c.kwargs["settled_count"] for c in mock_send.call_args_list]

    def test_notifies_only_main_and_the_retry_that_settled(self):
        """やり直しの5回のうち1回だけで約定した日は、9:05 の分とその1回の分だけ知らせる"""
        order = {"symbol": "7203", "qty": 100}
        # 9:05 は当日の行が無く0件、9:15・9:25 も0件、9:35 で1件約定、その後は約定待ちが無い
        settled_per_run = [[], [], [], [order]]
        pending_per_retry = [1, 1, 1, 0, 0]
        counts = self._run_morning(settled_per_run, pending_per_retry)
        self.assertEqual(counts, [0, 1])

    def test_no_pending_day_notifies_main_only(self):
        """約定待ちが無い日は、今までどおり 9:05 の分だけ知らせる"""
        order = {"symbol": "7203", "qty": 100}
        counts = self._run_morning([[order]], [0, 0, 0, 0, 0])
        self.assertEqual(counts, [1])


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
