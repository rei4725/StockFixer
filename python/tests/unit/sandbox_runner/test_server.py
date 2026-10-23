import json
import subprocess

from sandbox_runner.job import RunnerConfig
from sandbox_runner.server import handle_run

from .test_job import JOB

CFG = RunnerConfig("stockfixer:2.22.0", "vol", "1g", "1", 120)


def test_runs_the_built_command_and_returns_its_output():
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["kw"] = cmd, kw
        return subprocess.CompletedProcess(cmd, 0, '{"status": "ok"}\n', "")

    status, out = handle_run(json.dumps(JOB).encode(), CFG, run=fake_run)
    assert status == 200 and out == {"returncode": 0, "stdout": '{"status": "ok"}\n', "stderr": ""}
    assert seen["cmd"][:2] == ["docker", "run"] and seen["kw"]["timeout"] == 120
    assert "shell" not in seen["kw"]


def test_bad_job_is_400_and_nothing_runs():
    calls = []
    status, out = handle_run(
        json.dumps({**JOB, "source_rel": "../x"}).encode(), CFG, run=lambda *a, **k: calls.append(1)
    )
    assert status == 400 and "source_rel" in out["error"] and calls == []


def test_broken_json_is_400():
    status, out = handle_run(b"{nope", CFG, run=lambda *a, **k: None)
    assert status == 400


def test_timeout_is_reported():
    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 120)

    assert handle_run(json.dumps(JOB).encode(), CFG, run=slow) == (200, {"timeout": True})
