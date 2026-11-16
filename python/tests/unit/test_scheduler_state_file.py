"""scheduler_state_file の関数を直に呼ぶテスト（#1850）。"""

from __future__ import annotations

import json
import logging
import os
import re

import pytest

from src.orchestration import scheduler_state_file
from src.orchestration.scheduler_state_file import (
    quarantine_state_file,
    read_state_file,
    write_state_file,
)


def _write(path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def test_read_missing_file_returns_empty_events(tmp_path):
    result = read_state_file(str(tmp_path / "state.json"))
    assert result.state == {"events": []}
    assert result.corrupt_detail is None


def test_read_valid_file_returns_state(tmp_path):
    path = tmp_path / "state.json"
    _write(path, json.dumps({"events": [{"job_id": "daily"}], "other": 1}))
    result = read_state_file(str(path))
    assert result.state == {"events": [{"job_id": "daily"}], "other": 1}
    assert result.corrupt_detail is None


def test_read_fills_events_when_not_list(tmp_path):
    path = tmp_path / "state.json"
    _write(path, json.dumps({"events": "broken"}))
    result = read_state_file(str(path))
    assert result.state == {"events": []}
    assert result.corrupt_detail is None


def test_read_broken_json_reports_corruption(tmp_path):
    path = tmp_path / "state.json"
    _write(path, "{not json")
    result = read_state_file(str(path))
    assert result.state is None
    assert result.corrupt_detail
    # 読むだけでは退避しない
    assert os.path.exists(path)


def test_read_top_level_not_dict_reports_corruption(tmp_path):
    path = tmp_path / "state.json"
    _write(path, json.dumps([1, 2]))
    result = read_state_file(str(path))
    assert result.state is None
    assert result.corrupt_detail == "最上位が dict ではない: list"


def test_quarantine_moves_file_with_timestamp(tmp_path, caplog):
    path = tmp_path / "state.json"
    _write(path, "{not json")
    with caplog.at_level(logging.ERROR, logger="scheduler"):
        moved = quarantine_state_file(str(path), "壊れた")
    assert moved is not None
    assert not os.path.exists(path)
    assert os.path.exists(moved)
    assert re.fullmatch(re.escape(str(path)) + r"\.corrupt-\d{8}T\d{12}Z", moved)
    assert "退避し、補完実行を止めます" in caplog.text
    assert f"壊れた 退避先={moved}" in caplog.text


def test_quarantine_failure_returns_none(tmp_path, caplog):
    path = tmp_path / "missing.json"
    with caplog.at_level(logging.ERROR, logger="scheduler"):
        moved = quarantine_state_file(str(path), "壊れた")
    assert moved is None
    assert "壊れたスケジューラ状態ファイルの退避に失敗" in caplog.text


def test_write_then_read_round_trip(tmp_path):
    path = tmp_path / "sub" / "state.json"
    state = {"events": [{"job_id": "日次"}]}
    write_state_file(str(path), state)
    with open(path, encoding="utf-8") as file:
        text = file.read()
    assert text == json.dumps(state, ensure_ascii=False, indent=2)
    assert read_state_file(str(path)).state == state
    assert [p.name for p in (tmp_path / "sub").iterdir()] == ["state.json"]


def test_write_failure_leaves_no_temp_file_and_keeps_old(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    _write(path, json.dumps({"events": []}))

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(scheduler_state_file.os, "fsync", _boom)
    with pytest.raises(OSError, match="disk full"):
        write_state_file(str(path), {"events": [{"job_id": "x"}]})

    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]
    with open(path, encoding="utf-8") as file:
        assert json.load(file) == {"events": []}
