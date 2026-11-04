"""例外の注文が残った回の約定のジョブの失敗のテスト（#1565）"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.orchestration import scheduler_queue
from src.orchestration.job_errors import NonRetryableJobError
from src.orchestration.jobs import settle_retry
from src.orchestration.scheduler_queue import SchedulerQueueManager

_JST = ZoneInfo("Asia/Tokyo")
_SETTLE_JOB = "daily_settle_orders"
_RETRY_JOB = "daily_settle_orders_retry"


def _manager(tmp_path, job_id, func=None):
    """本番の設定のままのジョブを持つ管理を返す。func を渡すと関数だけ替える"""
    from run_scheduler import SCHEDULE_CONFIG

    config = dict(SCHEDULE_CONFIG[job_id])
    if func is not None:
        config["func"] = func
    state_path = tmp_path / "state.json"
    notifier = MagicMock()
    manager = SchedulerQueueManager(
        {job_id: config}, state_file_path=str(state_path), error_notifier=notifier
    )
    return manager, state_path, notifier


def _last_status(state_path):
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    return [e["status"] for e in saved["events"]][-1]


def _run_settle_job(tmp_path, errors):
    """9:05 のジョブを run_job で流し、記録と各呼び出しのモックを返す"""
    manager, state_path, notifier = _manager(tmp_path, _SETTLE_JOB)
    with (
        patch("src.trading.brokers.paper.paper_broker.PaperBroker") as broker_cls,
        patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter"),
        patch("src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink"),
        patch("src.reporting.discord.discord_utils.send_daily_settle_completion"),
        patch("src.reporting.discord.discord_utils.send_webhook_notification") as webhook,
        patch.object(scheduler_queue, "ping_heartbeat") as ping,
        patch.object(scheduler_queue.time, "sleep") as sleep,
    ):
        broker_cls.return_value.settle_pending_orders.return_value = []
        broker_cls.return_value.last_settle_errors = errors
        raised = None
        try:
            manager.run_job(_SETTLE_JOB, reason="scheduled", period_key="2026-10-22")
        except NonRetryableJobError as exc:
            raised = exc
    settle = broker_cls.return_value.settle_pending_orders
    return raised, _last_status(state_path), settle, webhook, ping, sleep, notifier


def test_settle_job_with_error_fails_once_after_notifying(tmp_path):
    """例外が1件の 9:05 の回は error で終わり、約定の処理を呼び直さない"""
    raised, status, settle, webhook, ping, sleep, notifier = _run_settle_job(tmp_path, 1)
    assert raised is not None
    assert status == "error"
    ping.assert_called_once_with(_SETTLE_JOB, success=False)
    settle.assert_called_once_with()
    sleep.assert_not_called()
    webhook.assert_called_once()
    assert "1 件" in webhook.call_args.args[1]
    notifier.assert_called_once()


def test_settle_job_with_only_deferred_orders_stays_success(tmp_path):
    """例外が0件で当日の行の無い注文が残る 9:05 の回は success のまま"""
    raised, status, settle, webhook, ping, _, notifier = _run_settle_job(tmp_path, 0)
    assert raised is None
    assert status == "success"
    ping.assert_called_once_with(_SETTLE_JOB, success=True)
    settle.assert_called_once_with()
    webhook.assert_not_called()
    notifier.assert_not_called()


def _run_retry_job(tmp_path, now, pending_counts, errors):
    """やり直しのジョブを run_job で流し、記録と各呼び出しのモックを返す"""
    manager, state_path, notifier = _manager(
        tmp_path, _RETRY_JOB, lambda: settle_retry.run_daily_settle_retry(now=now)
    )
    with (
        patch.object(settle_retry, "count_pending_paper_orders", side_effect=pending_counts),
        patch.object(settle_retry, "run_daily_settle_orders", return_value=errors) as settle,
        patch("src.reporting.discord.discord_utils.send_webhook_notification") as webhook,
        patch.object(scheduler_queue, "ping_heartbeat") as ping,
    ):
        raised = None
        try:
            manager.run_job(_RETRY_JOB, reason="scheduled", period_key="2026-10-22")
        except NonRetryableJobError as exc:
            raised = exc
    return raised, _last_status(state_path), settle, webhook, ping, notifier


def test_retry_job_with_error_at_cutoff_fails_after_notifying(tmp_path):
    """締めの回に例外の注文が残ると、残りの知らせの後にジョブが失敗になる"""
    now = datetime(2026, 10, 22, 9, 55, tzinfo=_JST)
    raised, status, settle, webhook, ping, notifier = _run_retry_job(tmp_path, now, [2, 2], 1)
    assert raised is not None
    assert status == "error"
    ping.assert_called_once_with(_RETRY_JOB, success=False)
    settle.assert_called_once_with(notify_when_none=False)
    webhook.assert_called_once()
    assert "約定処理の例外 1 件" in webhook.call_args.args[1]
    notifier.assert_called_once()


@pytest.mark.parametrize("minute", [15, 25, 35, 45])
def test_retry_job_with_error_before_cutoff_stays_success(tmp_path, minute):
    """締めより前の回は、後の回が約定し直すので例外があっても失敗にしない"""
    now = datetime(2026, 10, 22, 9, minute, tzinfo=_JST)
    raised, status, _, webhook, ping, notifier = _run_retry_job(tmp_path, now, [1], 1)
    assert raised is None
    assert status == "success"
    ping.assert_called_once_with(_RETRY_JOB, success=True)
    webhook.assert_not_called()
    notifier.assert_not_called()


def test_retry_job_with_only_deferred_orders_at_cutoff_stays_success(tmp_path):
    """例外が0件で当日の行の無い注文が残る締めの回は success のまま"""
    now = datetime(2026, 10, 22, 9, 55, tzinfo=_JST)
    raised, status, _, webhook, ping, notifier = _run_retry_job(tmp_path, now, [1, 1], 0)
    assert raised is None
    assert status == "success"
    ping.assert_called_once_with(_RETRY_JOB, success=True)
    webhook.assert_called_once()
    notifier.assert_not_called()


def test_retry_job_with_nothing_left_at_cutoff_stays_success(tmp_path):
    """締めの回で約定待ちが残らなければ、例外の件数があっても失敗にしない"""
    now = datetime(2026, 10, 22, 9, 55, tzinfo=_JST)
    raised, status, _, webhook, _, notifier = _run_retry_job(tmp_path, now, [1, 0], 1)
    assert raised is None
    assert status == "success"
    webhook.assert_not_called()
    notifier.assert_not_called()
