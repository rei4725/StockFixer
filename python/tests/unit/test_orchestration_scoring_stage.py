"""src.orchestration.jobs.scoring_stage のテスト"""

from datetime import date
from unittest.mock import MagicMock

import pytest

from src.orchestration.jobs import scoring_stage as mod
from src.orchestration.types import PipelineStage

D = date(2026, 11, 4)  # ISO 2026 週45


@pytest.fixture
def handler(monkeypatch):
    h = MagicMock()
    monkeypatch.setattr(mod, "_handle_stage_error", h)
    return h


def _patch(monkeypatch, target, name, **kw):
    m = MagicMock(**kw)
    monkeypatch.setattr(f"src.scoring.{target}.{name}", m)
    return m


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
