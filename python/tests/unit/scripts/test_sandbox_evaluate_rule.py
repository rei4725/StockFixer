from __future__ import annotations

import json
import subprocess

import pandas as pd
import scripts.sandbox_evaluate_rule as sandbox_script

from src.backtest.sandbox_executor import run_sandboxed_evaluation
from src.backtest.types import FactoryEvaluation, FactoryHypothesis


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


def test_main_reports_crash_in_generate_signal_as_repairable(monkeypatch, tmp_path, capsys):
    """evaluate_hypothesis()は銘柄ごとの例外を握りつぶすため、generate_signal()の
    クラッシュをスモークテストで先回りして検出できることを確認する（Task 7.5）。
    このスモークテストが無いと、クラッシュは「取引数0でゲート不合格」に埋もれてしまう。
    """
    monkeypatch.setenv("STOCKFIXER_SANDBOX", "1")

    source_file = tmp_path / "candidate.py"
    source_file.write_text(
        "class CrashingRule:\n"
        "    name = 'crashing_rule'\n"
        "    description = 'test'\n"
        "    def generate_signal(self, df):\n"
        "        raise ValueError('boom')\n",
        encoding="utf-8",
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    df = pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1_000_000},
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
            "CrashingRule",
            "--rule-name",
            "crashing_rule",
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
    assert rc == 1
    out = json.loads(capsys.readouterr().out.strip())
    assert out["status"] == "error"
    assert out["error_type"] == "ValueError"
    assert "boom" in out["traceback"]


def test_new_aggregate_fields_survive_sandbox_round_trip(monkeypatch, tmp_path, capsys):
    """n_symbols_with_signal / n_effective_symbols / avg_trades_per_symbol が
    サンドボックス往復（sandbox_evaluate_rule.py のペイロード構築 →
    sandbox_executor.py の FactoryEvaluation 再構築）で欠落しないことを確認する（#625）。

    どちらか片方でも新フィールドの受け渡しを書き忘れると、このテストは失敗する
    （キーの存在だけでなく、区別可能な非デフォルト値そのものを検証しているため）。
    """
    monkeypatch.setenv("STOCKFIXER_SANDBOX", "1")

    # --- producer側: evaluate_hypothesis を既知の値を持つ FactoryEvaluation に差し替える ---
    fake_hypothesis = FactoryHypothesis(
        rule_spec={
            "type": "generated_code",
            "source_code": "class X:\n    pass\n",
            "class_name": "X",
            "rule_name": "x",
            "description": "x",
        },
        market="us",
    )
    fake_evaluation = FactoryEvaluation(
        hypothesis=fake_hypothesis,
        sharpe_ratio=1.23,
        sharpe_per_trade=0.12,
        win_rate=0.5,
        num_trades=42,
        max_drawdown=-0.1,
        total_return=0.2,
        window_returns=[0.01, 0.02],
        n_symbols=3,
        n_symbols_with_signal=3,
        n_effective_symbols=2,
        avg_trades_per_symbol=21.0,
    )
    monkeypatch.setattr(
        "src.backtest.factory.evaluate_hypothesis", lambda *args, **kwargs: fake_evaluation
    )

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
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1_000_000},
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
    stdout = capsys.readouterr().out.strip()
    payload = json.loads(stdout)
    assert payload["evaluation"]["n_symbols_with_signal"] == 3
    assert payload["evaluation"]["n_effective_symbols"] == 2
    assert payload["evaluation"]["avg_trades_per_symbol"] == 21.0

    # --- consumer側: docker run の標準出力として同じ JSON を返させ、
    #     run_sandboxed_evaluation が正しく FactoryEvaluation に復元することを確認する ---
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
    monkeypatch.setattr(
        "src.backtest.sandbox_executor.subprocess.run", lambda *args, **kwargs: fake_proc
    )

    result = run_sandboxed_evaluation(fake_hypothesis, "/dummy_data", "/dummy_windows.json")

    assert result.kind == "gate_evaluated"
    assert result.evaluation is not None
    assert result.evaluation.n_symbols_with_signal == 3
    assert result.evaluation.n_effective_symbols == 2
    assert result.evaluation.avg_trades_per_symbol == 21.0


def test_old_sandbox_payload_without_new_fields_falls_back_to_defaults(monkeypatch):
    """新フィールドを含まない旧サンドボックスイメージ由来のペイロードでも
    KeyError にならず、FactoryEvaluation の既定値にフォールバックすることを確認する。
    FACTORY_SANDBOX_IMAGE は固定イメージのため、ホストより古いコンテナが
    稼働し続ける状況が起こり得る（#625）。
    """
    fake_hypothesis = FactoryHypothesis(
        rule_spec={
            "type": "generated_code",
            "source_code": "class X:\n    pass\n",
            "class_name": "X",
            "rule_name": "x",
            "description": "x",
        },
        market="us",
    )
    old_payload = {
        "status": "ok",
        "evaluation": {
            "sharpe_ratio": 1.0,
            "sharpe_per_trade": 0.1,
            "win_rate": 0.5,
            "num_trades": 10,
            "max_drawdown": -0.05,
            "total_return": 0.1,
            "window_returns": [0.01],
            "n_symbols": 1,
            # n_symbols_with_signal / n_effective_symbols / avg_trades_per_symbol は
            # 意図的に含めない（旧イメージを模擬）。
        },
    }
    fake_proc = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=json.dumps(old_payload), stderr=""
    )
    monkeypatch.setattr(
        "src.backtest.sandbox_executor.subprocess.run", lambda *args, **kwargs: fake_proc
    )

    result = run_sandboxed_evaluation(fake_hypothesis, "/dummy_data", "/dummy_windows.json")

    assert result.kind == "gate_evaluated"
    assert result.evaluation is not None
    assert result.evaluation.n_symbols_with_signal == 0
    assert result.evaluation.n_effective_symbols == 0
    assert result.evaluation.avg_trades_per_symbol == 0.0
