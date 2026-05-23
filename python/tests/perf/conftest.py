"""
パフォーマンスベンチマーク用 conftest

セッション終了時に benchmark_results.json を書き出す。
"""

import json
import os

import pytest

_bench: dict = {}


@pytest.fixture
def record_benchmark():
    def _record(name: str, elapsed_s: float) -> None:
        _bench[name] = round(elapsed_s, 4)

    return _record


def pytest_sessionfinish(session, exitstatus):
    if _bench:
        out = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
        with open(out, "w") as f:
            json.dump(_bench, f, indent=2)
