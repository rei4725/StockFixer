"""run_promotion_gate_trend.py（昇格ゲートの流れの表の CLI）のテスト（#2073）。"""

import json
import sys

import pandas as pd


def _result(evaluated_at, slippage_passed):
    return {
        "shadow_model": "challenger",
        "current_model": "champion",
        "evaluated_at": evaluated_at,
        "eligible": slippage_passed,
        "criteria": {
            "hit_rate": {"shadow": 0.55, "floor": 0.5, "passed": True},
            "slippage": {"shadow": 0.012, "current": 0.010, "passed": slippage_passed},
        },
        "wf_source_file": "wf_summary.csv",
    }


def _write_two(directory):
    for name, at, passed in [
        ("20261109_030000_promotion_result.json", "2026-11-09T03:00:00", False),
        ("20261116_030000_promotion_result.json", "2026-11-16T03:00:00", False),
    ]:
        (directory / name).write_text(json.dumps(_result(at, passed)), encoding="utf-8")


def test_cli_markdown_to_stdout(tmp_path, monkeypatch, capsys):
    import run_promotion_gate_trend as cli

    _write_two(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_promotion_gate_trend.py", "--dir", str(tmp_path)])
    cli.main()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("| file | ")
    assert "slippage_fail_streak" in lines[0]
    assert len(lines) == 4
    assert "20261109_030000_promotion_result" in lines[2]
    assert "null" in lines[2]


def test_cli_csv_tail_keeps_streak_counted_over_all_runs(tmp_path, monkeypatch):
    import run_promotion_gate_trend as cli

    _write_two(tmp_path)
    out = tmp_path / "out.csv"
    argv = ["run_promotion_gate_trend.py", "--dir", str(tmp_path), "--tail", "1"]
    argv += ["--format", "csv", "--output", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    df = pd.read_csv(out)
    assert df["file"].tolist() == ["20261116_030000_promotion_result"]
    assert df["slippage_fail_streak"].tolist() == [2]


def test_cli_empty_dir_says_no_data(tmp_path, monkeypatch, capsys):
    import run_promotion_gate_trend as cli

    monkeypatch.setattr(sys, "argv", ["run_promotion_gate_trend.py", "--dir", str(tmp_path)])
    cli.main()
    assert capsys.readouterr().out == "データなし" + chr(10)
