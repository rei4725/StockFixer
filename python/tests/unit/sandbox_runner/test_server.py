import json
import subprocess

from sandbox_runner import server
from sandbox_runner.job import RunnerConfig
from sandbox_runner.server import handle_run, parse_length

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
    assert seen["cmd"][2] == "--name" and seen["cmd"][3].startswith("factory-sandbox-")
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


def test_non_utf8_body_is_400():
    calls = []
    status, out = handle_run(b"\xff\xfe", CFG, run=lambda *a, **k: calls.append(1))
    assert status == 400 and "error" in out and calls == []


def test_timeout_is_reported_and_the_container_is_killed():
    calls = []

    def slow(cmd, **kw):
        calls.append(cmd)
        if cmd[1] == "run":
            raise subprocess.TimeoutExpired(cmd, 120)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    assert handle_run(json.dumps(JOB).encode(), CFG, run=slow) == (200, {"timeout": True})
    name = calls[0][calls[0].index("--name") + 1]
    assert calls[1] == ["docker", "kill", name]


def test_kill_errors_are_ignored_after_timeout():
    def slow(cmd, **kw):
        if cmd[1] == "run":
            raise subprocess.TimeoutExpired(cmd, 120)
        raise OSError("docker gone")

    assert handle_run(json.dumps(JOB).encode(), CFG, run=slow) == (200, {"timeout": True})


def test_missing_docker_is_500():
    def missing(cmd, **kw):
        raise FileNotFoundError("docker")

    status, out = handle_run(json.dumps(JOB).encode(), CFG, run=missing)
    assert status == 500 and out["error"].startswith("docker を実行できません")


def test_parse_length():
    assert parse_length(None) == 0
    assert parse_length("") == 0
    assert parse_length("12") == 12
    assert parse_length("abc") is None
    assert parse_length("-1") is None


def test_busy_runner_is_503_and_nothing_runs():
    calls = []
    held = [server._SLOTS.acquire(blocking=False) for _ in range(2)]
    try:
        assert all(held)
        status, out = handle_run(json.dumps(JOB).encode(), CFG, run=lambda *a, **k: calls.append(1))
    finally:
        for _ in held:
            server._SLOTS.release()
    assert status == 503 and "混んでいます" in out["error"] and calls == []


def test_slot_is_released_after_a_run():
    def boom(cmd, **kw):
        raise FileNotFoundError("docker")

    for _ in range(5):
        status, _ = handle_run(json.dumps(JOB).encode(), CFG, run=boom)
        assert status == 500
