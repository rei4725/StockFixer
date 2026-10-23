import pytest
from sandbox_runner.job import JobError, RunnerConfig, build_command

CFG = RunnerConfig(
    image="stockfixer:2.22.0",
    volume="stockfixer_factory_sandbox",
    memory="1g",
    cpus="1",
    timeout_seconds=120,
)
JOB = {
    "source_rel": "factory_sandbox_src_ab12/candidate.py",
    "data_rel": "factory_sandbox_data_cd34",
    "windows_rel": "factory_sandbox_windows_ef56.json",
    "class_name": "SafeRule",
    "rule_name": "safe_rule",
    "description": "test rule",
    "market": "us",
    "lookback_years": 3,
    "min_trades_per_symbol": 3,
}


def test_builds_the_fixed_sandbox_command():
    cmd = build_command(JOB, CFG)
    assert cmd[:2] == ["docker", "run"]
    for flag in ("--rm", "--read-only"):
        assert flag in cmd
    assert cmd[cmd.index("--pull") + 1] == "never"
    assert cmd[cmd.index("--pids-limit") + 1] == "256"
    assert cmd[cmd.index("--cap-drop") + 1] == "ALL"
    assert cmd[cmd.index("--security-opt") + 1] == "no-new-privileges"
    assert cmd.index("--security-opt") < cmd.index("stockfixer:2.22.0")
    assert cmd[cmd.index("--network") + 1] == "none"
    assert cmd[cmd.index("--memory") + 1] == "1g"
    assert cmd[cmd.index("--cpus") + 1] == "1"
    assert "stockfixer_factory_sandbox:/sandbox_share:ro" in cmd
    assert "FACTORY_GATE_MIN_TRADES_PER_SYMBOL=3" in cmd
    i = cmd.index("stockfixer:2.22.0")
    assert cmd[i + 1 : i + 3] == ["python", "scripts/sandbox_evaluate_rule.py"]
    assert (
        cmd[cmd.index("--source-file") + 1]
        == "/sandbox_share/factory_sandbox_src_ab12/candidate.py"
    )
    assert "--rule-name=safe_rule" in cmd
    assert "--description=test rule" in cmd
    assert cmd[cmd.index("--data-dir") + 1] == "/sandbox_share/factory_sandbox_data_cd34"
    assert (
        cmd[cmd.index("--windows-file") + 1] == "/sandbox_share/factory_sandbox_windows_ef56.json"
    )


@pytest.mark.parametrize(
    "key, value",
    [
        ("source_rel", "../etc/passwd"),
        ("source_rel", "/etc/passwd"),
        ("data_rel", "a/../../b"),
        ("windows_rel", "C:/x.json"),
        ("source_rel", ""),
    ],
)
def test_rejects_paths_outside_the_share(key, value):
    with pytest.raises(JobError):
        build_command({**JOB, key: value}, CFG)


@pytest.mark.parametrize(
    "key, value",
    [
        ("class_name", "Bad Name"),
        ("class_name", "--privileged"),
        ("market", "jp; rm"),
        ("market", "eu"),
        ("lookback_years", 0),
        ("lookback_years", 31),
        ("lookback_years", "3"),
        ("min_trades_per_symbol", -1),
        ("description", "x" * 4001),
        ("description", "a\nb"),
        ("description", "a\rb"),
        ("description", "a\x00b"),
        ("description", 5),
        ("rule_name", ""),
        ("rule_name", "x" * 201),
        ("rule_name", "a\nb"),
        ("rule_name", "a\x00b"),
        ("rule_name", "ok\n"),
        ("class_name", "Ok\n"),
        ("rule_name", 5),
    ],
)
def test_rejects_bad_arguments(key, value):
    with pytest.raises(JobError):
        build_command({**JOB, key: value}, CFG)


@pytest.mark.parametrize("name", ["RSI Divergence", "rsi-divergence", "-v", "x" * 200])
def test_rule_name_accepts_free_text_as_one_argv_item(name):
    cmd = build_command({**JOB, "rule_name": name}, CFG)
    assert f"--rule-name={name}" in cmd


def test_description_accepts_leading_dash_and_long_text():
    assert "--description=-x" in build_command({**JOB, "description": "-x"}, CFG)
    long_text = "説明" * 2000
    assert f"--description={long_text}" in build_command({**JOB, "description": long_text}, CFG)
    assert "--description=" in build_command({**JOB, "description": ""}, CFG)


def test_rejects_unknown_or_missing_keys():
    with pytest.raises(JobError):
        build_command({**JOB, "image": "evil:latest"}, CFG)
    with pytest.raises(JobError):
        build_command({k: v for k, v in JOB.items() if k != "market"}, CFG)


def test_config_reads_env(monkeypatch):
    monkeypatch.setenv("SANDBOX_IMAGE", "stockfixer:9.9.9")
    monkeypatch.setenv("SANDBOX_SHARE_VOLUME", "vol")
    monkeypatch.delenv("SANDBOX_MEMORY", raising=False)
    monkeypatch.delenv("SANDBOX_CPUS", raising=False)
    monkeypatch.delenv("SANDBOX_TIMEOUT_SECONDS", raising=False)
    assert RunnerConfig.from_env() == RunnerConfig("stockfixer:9.9.9", "vol", "1g", "1", 120)


def test_config_requires_image_and_volume(monkeypatch):
    monkeypatch.delenv("SANDBOX_IMAGE", raising=False)
    monkeypatch.setenv("SANDBOX_SHARE_VOLUME", "vol")
    with pytest.raises(RuntimeError):
        RunnerConfig.from_env()
