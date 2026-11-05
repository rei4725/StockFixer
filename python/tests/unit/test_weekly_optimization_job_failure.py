"""週の最適化で結果が空か失敗が線を超えた回をジョブの失敗にし、同じ週に補完する（#1581）"""

import json
import os
import sys
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.orchestration.job_errors import NonRetryableJobError  # noqa: E402
from src.orchestration.jobs.weekly import run_weekly_optimization  # noqa: E402
from src.orchestration.scheduler_queue import SchedulerQueueManager  # noqa: E402

_JOB_ID = "weekly_optimization"
# 2026-11-04 は水曜で 2026-W45。定時は 01:00
_WEEK_KEY = "2026-W45"
_SUCCESS = {"market": "jp", "symbol": "7203", "status": "success"}
_SKIPPED = {"market": "jp", "symbol": "6758", "status": "skipped"}
_ERROR = {"market": "jp", "symbol": "9984", "status": "error", "error": "x"}


class _Mocks:
    def __init__(self, batch: MagicMock, completion: MagicMock, heartbeat: MagicMock) -> None:
        self.batch = batch
        self.completion = completion
        self.heartbeat = heartbeat


@pytest.fixture(name="mocks")
def _mocks(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Mocks]:
    monkeypatch.delenv("USE_OPTUNA", raising=False)
    with (
        patch("src.watchlist.batch_runner.load_target_symbols", return_value=[]),
        patch("src.backtest.optimizer.run_optimize_batch") as batch,
        patch("src.reporting.discord.discord_utils.send_optimization_completion") as completion,
        patch("src.orchestration.scheduler_queue.ping_heartbeat") as heartbeat,
    ):
        yield _Mocks(batch, completion, heartbeat)


def _manager(tmp_path, now: datetime | None = None) -> SchedulerQueueManager:
    """本番の設定の週の最適化を、本物の run_weekly_optimization で動かす manager を作る"""
    from run_scheduler import SCHEDULE_CONFIG

    config = {
        _JOB_ID: {
            **SCHEDULE_CONFIG[_JOB_ID],
            "func": run_weekly_optimization,
            "max_retries": 2,
            "retry_base_wait_seconds": 0.0,
        }
    }
    manager = SchedulerQueueManager(config, state_file_path=str(tmp_path / "state.json"))
    fixed_now = now or datetime(2026, 11, 4, 1, 0, tzinfo=manager.tz)
    manager.now = lambda: fixed_now  # type: ignore[method-assign]
    return manager


def _events(tmp_path) -> list[dict[str, Any]]:
    text = (tmp_path / "state.json").read_text(encoding="utf-8")
    return json.loads(text)["events"]


def _finished(tmp_path, job_id: str = _JOB_ID) -> list[dict[str, Any]]:
    return [e for e in _events(tmp_path) if e["job_id"] == job_id and e["status"] != "started"]


@pytest.mark.parametrize(
    "results, success, failed, skipped",
    [
        pytest.param([_ERROR] * 3, 0, 3, 0, id="all-error"),
        pytest.param([_SUCCESS, _ERROR, _ERROR], 1, 2, 0, id="over-line"),
        pytest.param([_SKIPPED, _ERROR, _ERROR, _ERROR], 0, 3, 1, id="over-line-with-skip"),
        pytest.param([], 0, 0, 0, id="empty"),
    ],
)
def test_failed_run_is_error_without_retry_after_notice(
    tmp_path, mocks, results, success, failed, skipped
):
    """結果が空か失敗が線を超えた回は、知らせの後に投げ、error と記録し、やり直さないこと"""
    mocks.batch.return_value = results
    manager = _manager(tmp_path)

    with pytest.raises(NonRetryableJobError):
        manager.run_job(_JOB_ID, reason="scheduled")

    mocks.completion.assert_called_once_with(success=success, failed=failed, skipped=skipped)
    mocks.batch.assert_called_once()
    assert [e["status"] for e in _finished(tmp_path)] == ["error"]
    mocks.heartbeat.assert_called_once_with(_JOB_ID, success=False)


def test_failed_run_still_raises_when_notice_fails(tmp_path, mocks):
    """完了の知らせが落ちても、失敗の回は error と記録すること"""
    mocks.batch.return_value = [_ERROR]
    mocks.completion.side_effect = RuntimeError("Discord に届かない")

    with pytest.raises(NonRetryableJobError):
        _manager(tmp_path).run_job(_JOB_ID, reason="scheduled")

    assert [e["status"] for e in _finished(tmp_path)] == ["error"]


@pytest.mark.parametrize(
    "results",
    [
        pytest.param([_SUCCESS, _SKIPPED, _ERROR], id="one-of-three"),
        pytest.param([_SUCCESS, _ERROR], id="exactly-on-line"),
        pytest.param([_SKIPPED, _SKIPPED, _SKIPPED, _ERROR, _ERROR], id="skip-in-denominator"),
    ],
)
def test_failures_under_line_stay_success(tmp_path, mocks, results):
    """線以下の失敗は投げず、skipped は分母に残し、success のままであること"""
    mocks.batch.return_value = results
    manager = _manager(tmp_path)

    assert manager.run_job(_JOB_ID, reason="scheduled") is True

    assert [e["status"] for e in _finished(tmp_path)] == ["success"]
    mocks.heartbeat.assert_called_once_with(_JOB_ID, success=True)


@pytest.mark.parametrize(
    "results",
    [
        pytest.param([_SUCCESS, _SUCCESS], id="all-success"),
        pytest.param([_SKIPPED, _SKIPPED], id="all-skipped"),
        pytest.param([_SUCCESS, _SKIPPED], id="success-and-skipped"),
    ],
)
def test_success_or_skip_only_run_stays_success(tmp_path, mocks, results):
    """全部の銘柄が成功かスキップの回は、今までどおり success で終わること"""
    mocks.batch.return_value = results

    assert _manager(tmp_path).run_job(_JOB_ID, reason="scheduled") is True

    assert [e["status"] for e in _finished(tmp_path)] == ["success"]


def test_failed_run_is_recovered_in_same_week_and_success_stops_it(tmp_path, mocks):
    """失敗の回の後に同じ週の補完が1回通り、その成功で止まること"""
    mocks.batch.return_value = [_ERROR, _ERROR]
    first = _manager(tmp_path)
    with pytest.raises(NonRetryableJobError):
        first.run_job(_JOB_ID, reason="scheduled")

    # 木曜の昼も同じ週として補完する
    now = datetime(2026, 11, 5, 12, 0, tzinfo=first.tz)
    manager = _manager(tmp_path, now)
    assert manager.should_recover(_JOB_ID, now=now) is True

    mocks.batch.return_value = [_SKIPPED, _SUCCESS]
    assert manager.run_job(_JOB_ID, reason="recovery") is True

    assert [e["status"] for e in _finished(tmp_path)] == ["error", "success"]
    assert manager.should_recover(_JOB_ID, now=now) is False


def test_recovery_stops_after_second_failure(tmp_path, mocks):
    """補完の回も失敗したら、同じ週にそれ以上は補完しないこと"""
    mocks.batch.return_value = [_ERROR]
    first = _manager(tmp_path)
    now = datetime(2026, 11, 4, 2, 0, tzinfo=first.tz)
    manager = _manager(tmp_path, now)

    with pytest.raises(NonRetryableJobError):
        manager.run_job(_JOB_ID, reason="scheduled")
    later = now + timedelta(minutes=720)
    manager = _manager(tmp_path, later)
    assert manager.should_recover(_JOB_ID, now=later) is True
    with pytest.raises(NonRetryableJobError):
        manager.run_job(_JOB_ID, reason="recovery")

    assert manager.get_execution_count(_JOB_ID, _WEEK_KEY) == 2
    assert manager.should_recover(_JOB_ID, now=later + timedelta(minutes=720)) is False


def _interrupted(index: int) -> dict[str, Any]:
    return {
        "job_id": _JOB_ID,
        "reason": "scheduled",
        "status": "started",
        "period_key": _WEEK_KEY,
        "run_id": f"run-{index}",
        "boot_id": f"boot-{index}",
        "started_at": "2020-01-01T00:00:00+00:00",
    }


def _failed(index: int) -> dict[str, Any]:
    return {
        "job_id": _JOB_ID,
        "status": "error",
        "period_key": _WEEK_KEY,
        "run_id": f"failed-{index}",
    }


@pytest.mark.parametrize(
    "interrupted, failed, expected_count, expected_recover",
    [
        pytest.param(40, 0, 0, True, id="interrupted-only-use-the-forgiven-limit"),
        pytest.param(40, 1, 1, True, id="failure-counts-apart-from-interrupted"),
        pytest.param(40, 2, 2, False, id="two-failures-stop-recovery"),
        pytest.param(0, 2, 2, False, id="two-failures-without-interruption"),
        pytest.param(41, 1, 2, False, id="interrupted-over-limit-adds-to-failure"),
    ],
)
def test_interrupted_and_failed_runs_are_counted_apart(
    tmp_path, interrupted, failed, expected_count, expected_recover
):
    """消えた回は 40 回まで数えず、失敗の回は別に数えて補完の上限に当たること"""
    from run_scheduler import SCHEDULE_CONFIG

    assert SCHEDULE_CONFIG[_JOB_ID]["max_executions_per_period"] == 2
    assert SCHEDULE_CONFIG[_JOB_ID]["max_interrupted_reruns_per_period"] == 40
    events = [_interrupted(i) for i in range(interrupted)]
    events += [_failed(i) for i in range(failed)]
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"events": events}), encoding="utf-8")
    manager = SchedulerQueueManager(SCHEDULE_CONFIG, state_file_path=str(state_path))

    now = datetime(2026, 11, 4, 2, 0, tzinfo=manager.tz)
    assert manager.get_execution_count(_JOB_ID, _WEEK_KEY) == expected_count
    assert manager.should_recover(_JOB_ID, now=now) is expected_recover
