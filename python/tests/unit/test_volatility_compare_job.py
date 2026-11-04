"""週のジョブ weekly_volatility_sizing_compare（#1547）。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.orchestration.jobs.volatility_compare_job import run_weekly_volatility_sizing_compare
from src.orchestration.scheduler_queue import SchedulerQueueManager

_JOB_ID = "weekly_volatility_sizing_compare"
_COMPARE = "src.backtest.volatility_compare.run_volatility_sizing_compare"


def _run_job_function(tmp_path: Path, compare: MagicMock) -> None:
    sunday = datetime(2026, 11, 8, 4, 0, tzinfo=timezone.utc)
    with (
        patch(_COMPARE, compare),
        patch("src.watchlist.batch_runner.load_target_symbols", return_value=["task"]),
        patch("src.utils.data_path_utils.get_results_dir", return_value=str(tmp_path)),
        patch("src.utils.japan_time.now_jst", return_value=sunday),
    ):
        run_weekly_volatility_sizing_compare()


def test_job_writes_the_weekly_json(tmp_path: Path) -> None:
    compare = MagicMock(return_value={"markets": {"us": {"symbols": 1}}})

    _run_job_function(tmp_path, compare)

    path = tmp_path / "backtest_compare" / "volatility_sizing" / "2026-W45.json"
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["markets"]["us"]["symbols"] == 1
    assert written["week"] == "2026-W45"
    compare.assert_called_once_with(["task"], "2021-11-08", "2026-11-08")


def test_job_raises_on_failure_and_writes_nothing(tmp_path: Path) -> None:
    compare = MagicMock(side_effect=RuntimeError("0件"))

    with pytest.raises(RuntimeError):
        _run_job_function(tmp_path, compare)

    assert not (tmp_path / "backtest_compare").exists()


def _manager(tmp_path: Path, func: Any, notifier: Any = None) -> SchedulerQueueManager:
    from run_scheduler import SCHEDULE_CONFIG

    config = {_JOB_ID: {**SCHEDULE_CONFIG[_JOB_ID], "func": func}}
    return SchedulerQueueManager(
        config, state_file_path=str(tmp_path / "state.jsonl"), error_notifier=notifier
    )


def test_period_limit_is_respected(tmp_path: Path) -> None:
    func = MagicMock()
    manager = _manager(tmp_path, func)

    assert manager.run_job(_JOB_ID, reason="scheduled") is True
    assert manager.run_job(_JOB_ID, reason="recovery") is False

    assert func.call_count == 1


def test_failure_is_notified_without_a_retry(tmp_path: Path) -> None:
    func = MagicMock(side_effect=RuntimeError("boom"))
    notifier = MagicMock()
    manager = _manager(tmp_path, func, notifier)

    with pytest.raises(RuntimeError):
        manager.run_job(_JOB_ID, reason="scheduled")

    assert func.call_count == 1
    notifier.assert_called_once()


def test_schedule_avoids_the_order_hours_and_other_sunday_jobs() -> None:
    from run_scheduler import SCHEDULE_CONFIG, job_weekly_volatility_sizing_compare

    config = SCHEDULE_CONFIG[_JOB_ID]
    assert config["func"] is job_weekly_volatility_sizing_compare
    assert (config["day_of_week"], config["hour"], config["minute"]) == ("sun", 13, 0)
    assert config["period"] == "weekly"
    assert config["max_executions_per_period"] == 1
    for job_id, other in SCHEDULE_CONFIG.items():
        if job_id == _JOB_ID or "sun" not in str(other["day_of_week"]):
            continue
        assert (other["hour"], other["minute"]) != (13, 0), job_id


def test_job_function_goes_through_the_facade() -> None:
    from run_scheduler import job_weekly_volatility_sizing_compare

    with patch("src.orchestration.scheduler.run_weekly_volatility_sizing_compare") as run:
        job_weekly_volatility_sizing_compare()

    run.assert_called_once_with()
