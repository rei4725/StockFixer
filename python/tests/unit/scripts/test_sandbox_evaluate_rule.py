from __future__ import annotations

import json

import pandas as pd
import scripts.sandbox_evaluate_rule as sandbox_script


def test_main_rejects_without_sandbox_flag(monkeypatch, capsys):
    monkeypatch.delenv("STOCKFIXER_SANDBOX", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "sandbox_evaluate_rule.py",
            "--source-file",
            "dummy.py",
            "--class-name",
            "X",
            "--rule-name",
            "x",
            "--description",
            "x",
            "--market",
            "us",
            "--lookback-years",
            "2",
            "--data-dir",
            "dummy_dir",
            "--windows-file",
            "dummy.json",
        ],
    )
    rc = sandbox_script.main()
    assert rc == 1
    out = json.loads(capsys.readouterr().out.strip())
    assert out["status"] == "error"
    assert "STOCKFIXER_SANDBOX" in out["traceback"]


def test_main_reports_actual_exception_type_on_generic_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("STOCKFIXER_SANDBOX", "1")

    missing_source_file = tmp_path / "does_not_exist.py"

    monkeypatch.setattr(
        "sys.argv",
        [
            "sandbox_evaluate_rule.py",
            "--source-file",
            str(missing_source_file),
            "--class-name",
            "X",
            "--rule-name",
            "x",
            "--description",
            "x",
            "--market",
            "us",
            "--lookback-years",
            "2",
            "--data-dir",
            "dummy_dir",
            "--windows-file",
            "dummy.json",
        ],
    )
    rc = sandbox_script.main()
    assert rc == 1
    out = json.loads(capsys.readouterr().out.strip())
    assert out["status"] == "error"
    assert out["error_type"] == "FileNotFoundError"


def test_main_success_path(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("STOCKFIXER_SANDBOX", "1")

    source_file = tmp_path / "candidate.py"
    source_file.write_text(
        "class GeneratedTestRule:\n"
        "    name = 'generated_test_rule'\n"
        "    description = 'test'\n"
        "    def generate_signal(self, df):\n"
        "        import pandas as pd\n"
        "        return pd.Series(0, index=df.index)\n",
        encoding="utf-8",
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    df = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Volume": 1_000_000,
        },
        index=dates,
    )
    df.to_parquet(data_dir / "TEST.parquet")

    windows_file = tmp_path / "windows.json"
    windows_file.write_text(
        json.dumps([["2024-01-01", "2024-02-01"], ["2024-02-01", "2024-03-01"]]),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "sandbox_evaluate_rule.py",
            "--source-file",
            str(source_file),
            "--class-name",
            "GeneratedTestRule",
            "--rule-name",
            "generated_test_rule",
            "--description",
            "test",
            "--market",
            "us",
            "--lookback-years",
            "2",
            "--data-dir",
            str(data_dir),
            "--windows-file",
            str(windows_file),
        ],
    )
    rc = sandbox_script.main()
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["status"] == "ok"
    assert "sharpe_ratio" in out["evaluation"]
    assert out["evaluation"]["n_symbols"] == 1
