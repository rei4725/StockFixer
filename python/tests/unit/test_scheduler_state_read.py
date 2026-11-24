"""スケジューラ状態ファイルの読み（src/utils/scheduler_state_read.py）のテスト。"""

import ast
import json
from pathlib import Path

import pytest

from src.utils import health_check, scheduler_failure_count, scheduler_state_read
from src.utils.scheduler_state_read import (
    SCHEDULER_STATE_FILE_NAME,
    SchedulerStateUnreadableError,
    find_quarantined_state,
    read_scheduler_state,
)

SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / SCHEDULER_STATE_FILE_NAME


def test_error_type_is_shared():
    assert (
        health_check.SchedulerStateUnreadableError
        is scheduler_failure_count.SchedulerStateUnreadableError
    )
    assert (
        health_check.SchedulerStateUnreadableError
        is scheduler_state_read.SchedulerStateUnreadableError
    )


def test_definitions_are_single_in_src():
    errors = []
    finders = []
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.ClassDef) and node.name == "SchedulerStateUnreadableError":
                errors.append(path.relative_to(SRC).as_posix())
            if isinstance(node, ast.FunctionDef) and node.name.lstrip("_") == (
                "find_quarantined_state"
            ):
                finders.append(path.relative_to(SRC).as_posix())
    assert errors == ["utils/scheduler_state_read.py"]
    assert finders == ["utils/scheduler_state_read.py"]


def test_missing_file_returns_none(state_path):
    assert read_scheduler_state(str(state_path)) is None


def test_missing_directory_returns_none(tmp_path):
    assert read_scheduler_state(str(tmp_path / "none" / SCHEDULER_STATE_FILE_NAME)) is None


def test_callers_keep_their_empty_values(tmp_path, state_path):
    assert health_check.load_scheduler_last_runs(str(tmp_path)) == {}
    assert scheduler_failure_count._count_consecutive_pipeline_failures(str(state_path)) == 0


def test_broken_json_raises(state_path):
    state_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(SchedulerStateUnreadableError) as exc_info:
        read_scheduler_state(str(state_path))
    cause = exc_info.value.__cause__
    assert isinstance(cause, json.JSONDecodeError)
    assert str(exc_info.value) == f"状態ファイルが読めない: {cause}"


def test_invalid_utf8_raises(state_path):
    state_path.write_bytes(b'{"events": "\xff"}')
    with pytest.raises(SchedulerStateUnreadableError) as exc_info:
        read_scheduler_state(str(state_path))
    cause = exc_info.value.__cause__
    assert isinstance(cause, UnicodeDecodeError)
    assert str(exc_info.value) == f"状態ファイルが読めない: {cause}"


def test_top_level_list_raises(state_path):
    state_path.write_text("[]", encoding="utf-8")
    with pytest.raises(SchedulerStateUnreadableError) as exc_info:
        read_scheduler_state(str(state_path))
    assert str(exc_info.value) == "状態ファイルが読めない: 最上位が dict ではない: list"


def test_only_quarantined_raises_with_latest_name(tmp_path, state_path):
    older = f"{SCHEDULER_STATE_FILE_NAME}.corrupt-20260101"
    newer = f"{SCHEDULER_STATE_FILE_NAME}.corrupt-20260102"
    for name in (newer, older):
        (tmp_path / name).write_text("x", encoding="utf-8")
    with pytest.raises(SchedulerStateUnreadableError) as exc_info:
        read_scheduler_state(str(state_path))
    assert str(exc_info.value) == f"状態ファイルが退避されたまま本体が無い: {newer}"


def test_valid_dict_is_returned(tmp_path, state_path):
    state = {"events": [{"job_id": "daily_pipeline", "status": "success"}]}
    state_path.write_text(json.dumps(state), encoding="utf-8")
    # 本体が読めれば、退避の跡が残っていても投げない
    (tmp_path / f"{SCHEDULER_STATE_FILE_NAME}.corrupt-1").write_text("x", encoding="utf-8")
    assert read_scheduler_state(str(state_path)) == state


def test_quarantine_prefix_follows_given_name(tmp_path):
    (tmp_path / "other.json.corrupt-1").write_text("x", encoding="utf-8")
    assert find_quarantined_state(str(tmp_path / "other.json")) == "other.json.corrupt-1"
    assert find_quarantined_state(str(tmp_path / SCHEDULER_STATE_FILE_NAME)) is None


def test_relative_path_searches_current_directory(tmp_path, monkeypatch):
    name = f"{SCHEDULER_STATE_FILE_NAME}.corrupt-1"
    (tmp_path / name).write_text("x", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert find_quarantined_state(SCHEDULER_STATE_FILE_NAME) == name
    with pytest.raises(SchedulerStateUnreadableError):
        health_check.load_scheduler_last_runs(".")
