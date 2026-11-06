"""週のレポートのジョブが失敗の回を error と記録して補完を通すことのテスト（#1582）"""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.orchestration import scheduler_queue
from src.orchestration.job_errors import NonRetryableJobError
from src.orchestration.scheduler_queue import SchedulerQueueManager

_JOB_ID = "weekly_report"
_PERIOD = "2026-W43"
# 木曜 02:00 の定時と recovery_delay_minutes の 15 分を過ぎた時刻
_AFTER_DELAY = datetime(2026, 10, 22, 2, 20, tzinfo=ZoneInfo("Asia/Tokyo"))
# 本番の recovery_min_gap_minutes の 240 分
_GAP = timedelta(minutes=240)
_WEEK_DF = pd.DataFrame(
    [
        {
            "market": "jp",
            "symbol": "7203",
            "direction_accuracy": 0.6,
            "mean_abs_error": 0.01,
            "n_samples": 30,
        }
    ]
)


def _manager(tmp_path):
    """本番の設定のままのジョブを持つ管理を返す"""
    from run_scheduler import SCHEDULE_CONFIG

    notifier = MagicMock()
    state_path = tmp_path / "state.json"
    manager = SchedulerQueueManager(
        {_JOB_ID: dict(SCHEDULE_CONFIG[_JOB_ID])},
        state_file_path=str(state_path),
        error_notifier=notifier,
    )
    return SimpleNamespace(manager=manager, state_path=state_path, notifier=notifier)


def _statuses(job):
    saved = json.loads(job.state_path.read_text(encoding="utf-8"))
    return [e["status"] for e in saved["events"]]


def _run(
    job,
    *,
    report_error=None,
    miss_error=None,
    paper_error=None,
    notice_error=None,
    week_df=_WEEK_DF,
):
    """週のレポートのジョブを run_job で流し、例外と各段のモックを返す"""
    base = "src.reporting.discord.discord_utils"
    query = "src.infrastructure.persistence.analytics_query.PostgresAnalyticsQuery"
    with (
        patch("src.prediction.db.load_drift_summary", return_value=_WEEK_DF),
        patch("src.prediction.db.load_weekly_accuracy_summary", return_value=week_df),
        patch("src.prediction.db.save_weekly_accuracy_snapshot") as save,
        patch(query),
        patch("src.reporting.llm_review.generate_weekly_review", return_value=None),
        patch("src.infrastructure.llm.factory.get_text_review_port"),
        patch(f"{base}.send_weekly_report", side_effect=report_error) as report,
        patch("src.prediction.db.load_top_prediction_misses", side_effect=miss_error),
        patch("src.prediction.miss_analysis.run_miss_analysis_batch", return_value=[]),
        patch(f"{base}.send_miss_analysis_summary") as miss_summary,
        patch(
            "src.trading.paper_equity.write_paper_weekly_results", side_effect=paper_error
        ) as paper,
        patch(f"{base}.send_webhook_text", side_effect=notice_error) as notice,
        patch.object(scheduler_queue, "ping_heartbeat") as ping,
        patch.object(scheduler_queue.time, "sleep") as sleep,
    ):
        raised = None
        try:
            job.manager.run_job(_JOB_ID, reason="scheduled", period_key=_PERIOD)
        except NonRetryableJobError as exc:
            raised = exc
    return SimpleNamespace(
        raised=raised,
        save=save,
        report=report,
        miss_summary=miss_summary,
        paper=paper,
        notice=notice,
        ping=ping,
        sleep=sleep,
    )


def _assert_failed_once(job, run):
    """error と記録され、heartbeat が失敗を送り、すぐの呼び直しをしていない"""
    assert run.raised is not None
    assert _statuses(job)[-1] == "error"
    assert "retry" not in _statuses(job)
    run.ping.assert_called_once_with(_JOB_ID, success=False)
    run.sleep.assert_not_called()
    job.notifier.assert_called_once()


def test_body_failure_runs_later_stages_then_fails_once(tmp_path):
    """本体の段が落ちた回は後の段が走り、最後に error で終わる"""
    job = _manager(tmp_path)
    run = _run(job, report_error=RuntimeError("送信エラー"))

    _assert_failed_once(job, run)
    assert "週のレポート本体" in str(run.raised)
    run.report.assert_called_once()
    run.miss_summary.assert_called_once()
    run.paper.assert_called_once()
    run.notice.assert_called_once()
    assert "❌ 週次レポート生成失敗" in run.notice.call_args.args[0]


def test_paper_failure_fails_once_after_all_stages(tmp_path):
    """紙の週の成績が落ちた回は全部の段が走り、最後に error で終わる"""
    job = _manager(tmp_path)
    run = _run(job, paper_error=RuntimeError("書けない"))

    _assert_failed_once(job, run)
    assert "紙の週の成績" in str(run.raised)
    run.report.assert_called_once()
    run.miss_summary.assert_called_once()
    run.paper.assert_called_once()
    run.notice.assert_called_once()
    assert "❌ 紙の売買の週の成績の書き出し失敗" in run.notice.call_args.args[0]


def test_both_failures_are_named_in_the_error(tmp_path):
    """両方が落ちた回は、落ちた段の両方を例外に書く"""
    job = _manager(tmp_path)
    run = _run(job, report_error=RuntimeError("a"), paper_error=RuntimeError("b"))

    _assert_failed_once(job, run)
    assert "週のレポート本体" in str(run.raised)
    assert "紙の週の成績" in str(run.raised)
    assert run.notice.call_count == 2


def test_failed_run_is_recovered_once(tmp_path):
    """失敗の回の後は間を過ぎて補完が1回通り、2回目も失敗なら補完しない（#1689）"""
    job = _manager(tmp_path)
    with patch.object(job.manager, "now", return_value=_AFTER_DELAY):
        _run(job, paper_error=RuntimeError("書けない"))
    assert job.manager.should_recover(_JOB_ID, now=_AFTER_DELAY + _GAP) is True

    with patch.object(job.manager, "now", return_value=_AFTER_DELAY + _GAP):
        second = _run(job, paper_error=RuntimeError("書けない"))
    assert second.raised is not None
    assert job.manager.should_recover(_JOB_ID, now=_AFTER_DELAY + 2 * _GAP) is False


def test_recovery_success_ends_recovery(tmp_path):
    """補完の回が通れば success と記録し、それ以上補完しない"""
    job = _manager(tmp_path)
    _run(job, report_error=RuntimeError("送信エラー"))

    second = _run(job)
    assert second.raised is None
    assert _statuses(job)[-1] == "success"
    assert job.manager.should_recover(_JOB_ID, now=_AFTER_DELAY) is False


def test_miss_analysis_failure_alone_stays_success(tmp_path):
    """外れ原因分析だけが落ちた回は success で終わる"""
    job = _manager(tmp_path)
    run = _run(job, miss_error=RuntimeError("分析エラー"))

    assert run.raised is None
    assert _statuses(job)[-1] == "success"
    run.ping.assert_called_once_with(_JOB_ID, success=True)
    run.report.assert_called_once()
    run.miss_summary.assert_not_called()
    run.paper.assert_called_once()
    run.notice.assert_not_called()
    job.notifier.assert_not_called()


def test_all_stages_passing_stays_success(tmp_path):
    """全部の段が通った回は success で終わり、知らせを出さない"""
    job = _manager(tmp_path)
    run = _run(job)

    assert run.raised is None
    assert _statuses(job)[-1] == "success"
    run.ping.assert_called_once_with(_JOB_ID, success=True)
    run.save.assert_called_once()
    run.notice.assert_not_called()
    job.notifier.assert_not_called()


def test_empty_week_notifies_and_stays_success(tmp_path):
    """週の精度の記録が0行の週は知らせを出し、ジョブは success のまま進む"""
    job = _manager(tmp_path)
    run = _run(job, week_df=pd.DataFrame())

    assert run.raised is None
    assert _statuses(job)[-1] == "success"
    run.report.assert_called_once()
    run.paper.assert_called_once()
    run.notice.assert_called_once()
    assert "⚠️ 週の精度の記録が0行" in run.notice.call_args.args[0]


def test_empty_week_notice_failure_does_not_fail_the_report(tmp_path):
    """0行の知らせの送信が落ちても、レポートの段を失敗にしない"""
    job = _manager(tmp_path)
    run = _run(job, week_df=pd.DataFrame(), notice_error=RuntimeError("通知エラー"))

    assert run.raised is None
    assert _statuses(job)[-1] == "success"
    run.notice.assert_called_once()
    run.report.assert_called_once()


def test_failure_notice_failure_still_fails_the_job(tmp_path):
    """失敗の知らせの送信が落ちても、ジョブは同じ失敗で終わる"""
    job = _manager(tmp_path)
    run = _run(job, report_error=RuntimeError("a"), notice_error=RuntimeError("通知エラー"))

    _assert_failed_once(job, run)
    run.paper.assert_called_once()


@pytest.mark.parametrize("error", [RuntimeError("x"), OSError("disk")])
def test_paper_results_returns_false_on_failure(error):
    """紙の週の成績は例外を出さず、成否を返す"""
    from src.orchestration.jobs.weekly import run_weekly_paper_results

    with (
        patch("src.trading.paper_equity.write_paper_weekly_results", side_effect=error),
        patch("src.reporting.discord.discord_utils.send_webhook_text"),
    ):
        assert run_weekly_paper_results() is False


def test_paper_results_returns_true_on_success():
    """書き出しが終わった回は True を返す"""
    from src.orchestration.jobs.weekly import run_weekly_paper_results

    with patch("src.trading.paper_equity.write_paper_weekly_results", return_value=None):
        assert run_weekly_paper_results() is True
