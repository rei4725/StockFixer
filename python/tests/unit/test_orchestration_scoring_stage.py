"""src.orchestration.jobs.scoring_stage のテスト"""

from datetime import date
from unittest.mock import MagicMock

import pytest

from src.orchestration.jobs import scoring_stage as mod
from src.orchestration.jobs import weekly
from src.orchestration.types import PipelineStage

D = date(2026, 11, 4)  # ISO 2026 週45


@pytest.fixture
def notify(monkeypatch):
    n = MagicMock()
    monkeypatch.setattr(mod, "_notify_stage_failure", n, raising=False)
    return n


@pytest.fixture
def handler(monkeypatch, notify):
    h = MagicMock()
    monkeypatch.setattr(mod, "_handle_stage_error", h)
    return h


def _patch(monkeypatch, target, name, **kw):
    m = MagicMock(**kw)
    monkeypatch.setattr(f"src.scoring.{target}.{name}", m)
    return m


def _run_snapshot():
    mod.snapshot_predictions_stage()


def _run_daily():
    mod.daily_scoring_stage()


def _run_versions():
    mod.log_rule_signal_versions_stage(D, "jp")


def _run_weekly():
    mod.weekly_rule_scoring_stage("jp", D)


# 段を動かす関数、差し替える先（src.scoring の下）、段のラベル
STAGES = [
    pytest.param(_run_snapshot, "db", "snapshot_predictions", mod.SNAPSHOT_STAGE, id="snapshot"),
    pytest.param(
        _run_daily, "rule_report", "run_daily_scoring", mod.DAILY_SCORING_STAGE, id="daily"
    ),
    pytest.param(
        _run_versions, "db", "log_rule_signal_versions", mod.VERSIONS_STAGE, id="versions"
    ),
    pytest.param(_run_weekly, "db", "snapshot_rule_best", mod.WEEKLY_STAGE, id="weekly"),
]


@pytest.fixture
def stage_mocks(monkeypatch):
    """段が呼ぶ先のうち、段の失敗の元にしないものを差し替える。"""
    _patch(monkeypatch, "version", "read_app_version", return_value="2.0.0")
    _patch(monkeypatch, "rule_report", "write_weekly_rule_accuracy")


@pytest.mark.parametrize("run,target,name,label", STAGES)
def test_failure_notifies_once_after_handling(
    monkeypatch, handler, notify, stage_mocks, run, target, name, label
):
    """例外の回は、ログの段の後に1回だけ知らせ、例外を外に出さない。"""
    boom = RuntimeError("x")
    order = []
    handler.side_effect = lambda *a: order.append("handle")
    notify.side_effect = lambda *a: order.append("notify")
    _patch(monkeypatch, target, name, side_effect=boom)
    run()
    handler.assert_called_once_with(PipelineStage.NON_CRITICAL, label, boom)
    notify.assert_called_once_with(label, boom)
    assert order == ["handle", "notify"]


@pytest.mark.parametrize("run,target,name,label", STAGES)
def test_success_does_not_notify(
    monkeypatch, handler, notify, stage_mocks, run, target, name, label
):
    _patch(monkeypatch, target, name)
    run()
    notify.assert_not_called()


@pytest.mark.parametrize("run,target,name,label", STAGES)
def test_send_failure_stays_inside_stage(monkeypatch, stage_mocks, run, target, name, label):
    """知らせの送信そのものが落ちても、段は例外を外に出さない。"""
    send = MagicMock(side_effect=OSError("webhook down"))
    monkeypatch.setattr("src.reporting.discord.discord_utils.send_webhook_text", send)
    _patch(monkeypatch, target, name, side_effect=RuntimeError("x"))
    run()
    send.assert_called_once()
    assert label in send.call_args.args[0]


def test_weekly_job_stays_success_when_stage_fails(monkeypatch, notify):
    """段が落ちても errors が0なら、知らせを出しつつジョブは今までどおり成功で終わる。"""
    monkeypatch.setattr(weekly, "weekly_rule_scoring_stage", mod.weekly_rule_scoring_stage)
    summary = {"evaluated": 3, "effective": 2, "skipped": 1, "errors": 0}
    monkeypatch.setattr(
        "src.backtest.rule_selector.evaluate_all_symbols", MagicMock(return_value=summary)
    )
    completion = MagicMock()
    monkeypatch.setattr(
        "src.reporting.discord.discord_utils.send_rule_evaluation_completion", completion
    )
    _patch(monkeypatch, "db", "snapshot_rule_best", side_effect=RuntimeError("x"))
    weekly.run_weekly_rule_evaluation()
    notify.assert_called_once()
    assert notify.call_args.args[0] == mod.WEEKLY_STAGE
    completion.assert_called_once()


def test_snapshot_ok(monkeypatch, handler):
    snap = _patch(monkeypatch, "db", "snapshot_predictions", return_value=3)
    _patch(monkeypatch, "version", "read_app_version", return_value="2.0.0")
    mod.snapshot_predictions_stage()
    snap.assert_called_once_with("2.0.0")
    handler.assert_not_called()


def test_snapshot_failure(monkeypatch, handler):
    _patch(monkeypatch, "db", "snapshot_predictions", side_effect=RuntimeError("x"))
    _patch(monkeypatch, "version", "read_app_version", return_value=None)
    mod.snapshot_predictions_stage()
    assert handler.call_args.args[:2] == (PipelineStage.NON_CRITICAL, "[3.6/6] 予想の記録")


def test_daily_scoring_ok(monkeypatch, handler):
    run = _patch(monkeypatch, "rule_report", "run_daily_scoring")
    mod.daily_scoring_stage()
    run.assert_called_once_with()
    handler.assert_not_called()


def test_daily_scoring_failure(monkeypatch, handler):
    _patch(monkeypatch, "rule_report", "run_daily_scoring", side_effect=RuntimeError("x"))
    mod.daily_scoring_stage()
    assert handler.call_args.args[:2] == (
        PipelineStage.NON_CRITICAL,
        "[7/6] ルールシグナルの採点",
    )


def test_versions_ok(monkeypatch, handler):
    log = _patch(monkeypatch, "db", "log_rule_signal_versions", return_value=2)
    _patch(monkeypatch, "version", "read_app_version", return_value="2.0.0")
    mod.log_rule_signal_versions_stage(D, "jp")
    log.assert_called_once_with(D, "jp", "2.0.0")
    handler.assert_not_called()


def test_versions_failure(monkeypatch, handler):
    _patch(monkeypatch, "db", "log_rule_signal_versions", side_effect=RuntimeError("x"))
    _patch(monkeypatch, "version", "read_app_version", return_value=None)
    mod.log_rule_signal_versions_stage(D, "jp")
    assert handler.call_args.args[0] is PipelineStage.NON_CRITICAL


def test_weekly_ok(monkeypatch, handler):
    snap = _patch(monkeypatch, "db", "snapshot_rule_best")
    write = _patch(monkeypatch, "rule_report", "write_weekly_rule_accuracy")
    mod.weekly_rule_scoring_stage("jp", D)
    snap.assert_called_once_with(D, "jp")
    write.assert_called_once_with("jp", 2026, 45)
    handler.assert_not_called()


def test_weekly_failure_in_snapshot_and_in_write(monkeypatch, handler):
    _patch(monkeypatch, "db", "snapshot_rule_best", side_effect=RuntimeError("x"))
    _patch(monkeypatch, "rule_report", "write_weekly_rule_accuracy")
    mod.weekly_rule_scoring_stage("jp", D)
    assert handler.call_count == 1
    _patch(monkeypatch, "db", "snapshot_rule_best")
    _patch(monkeypatch, "rule_report", "write_weekly_rule_accuracy", side_effect=OSError("y"))
    mod.weekly_rule_scoring_stage("jp", D)
    assert handler.call_count == 2
