"""deploy_windows.json の形を確かめる。

deploy_window.ps1 が読む自動デプロイの見送りの窓の表が、JSON として読め、
どの窓も HH:mm の形で start < end、name が空でないことを確かめる (#2015)。
"""

import json
import re
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[3] / "deploy_windows.json"
_HHMM = re.compile(r"^([01][0-9]|2[0-4]):[0-5][0-9]$")


def _load() -> dict:
    if not _PATH.exists():
        pytest.skip("deploy_windows.json が無い環境")
    return json.loads(_PATH.read_text(encoding="utf-8"))


def test_windows_are_well_formed() -> None:
    windows = _load()["windows"]
    assert windows
    for w in windows:
        assert _HHMM.match(w["start"]), w
        assert _HHMM.match(w["end"]), w
        assert w["start"] < w["end"], w
        assert w["name"].strip(), w


def test_weekday_daily_pipeline_window_exists() -> None:
    windows = _load()["windows"]
    hits = [
        w
        for w in windows
        if w["start"] == "07:10" and w["end"] == "08:30" and "daily" not in w and "day" not in w
    ]
    assert len(hits) == 1
