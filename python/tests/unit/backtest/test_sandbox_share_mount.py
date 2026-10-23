"""サンドボックスへの入力の受け渡し（#757）。

本体がコンテナの中からホストの Docker でサンドボックスを起動する構成では、本体の /tmp の
パスをホストの Docker が解決できず、毎晩すべての Claude 生成候補が「入力が見つからない」で
全滅していた。共有ボリューム方式とホスト実行時の bind mount 方式の両方を固定する。
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.backtest import claude_rule_generator, sandbox_executor
from src.backtest.sandbox_executor import (
    SandboxRunResult,
    prepare_sandbox_data,
    run_sandboxed_evaluation,
)
from src.backtest.types import FactoryHypothesis

_SOURCE = (
    "import pandas as pd\n"
    "class ProbeRule:\n"
    "    name = 'probe'\n"
    "    description = 'probe'\n"
    "    def generate_signal(self, df):\n"
    "        return pd.Series(0, index=df.index)\n"
)
_HYPOTHESIS = FactoryHypothesis(
    rule_spec={
        "type": "generated_code",
        "source_code": _SOURCE,
        "class_name": "ProbeRule",
        "rule_name": "probe",
        "description": "probe",
    },
    market="us",
)


def _run_capturing_job(monkeypatch, reply: dict):
    """runner に送ったジョブを捕まえ、決めた答えを返す。"""
    captured: dict = {}

    def _fake_post(job):
        captured["job"] = job
        return reply

    monkeypatch.setattr(sandbox_executor, "_post_job", _fake_post)
    return captured


def _ok_payload() -> str:
    return json.dumps(
        {
            "status": "ok",
            "evaluation": {
                "sharpe_ratio": 0.0,
                "sharpe_per_trade": 0.0,
                "win_rate": 0.0,
                "num_trades": 0,
                "max_drawdown": 0.0,
                "total_return": 0.0,
                "window_returns": [],
                "n_symbols": 1,
            },
        }
    )


def _prepare(tmp_dir: str | None = None):
    df = pd.DataFrame({"Close": [1.0, 2.0]}, index=pd.bdate_range("2024-01-01", periods=2))
    windows = [(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))]
    return prepare_sandbox_data({"TEST": df}, windows)


class TestSharedVolumeMode:
    @pytest.fixture(autouse=True)
    def _share(self, monkeypatch, tmp_path):
        self.share_dir = tmp_path / "share"
        self.share_dir.mkdir()
        monkeypatch.setattr(sandbox_executor, "FACTORY_SANDBOX_SHARE_DIR", str(self.share_dir))
        monkeypatch.setattr(sandbox_executor, "FACTORY_SANDBOX_SHARE_VOLUME", "vol_share")

    def test_inputs_are_created_inside_the_shared_directory(self):
        data_dir, windows = _prepare()
        assert os.path.commonpath([data_dir, str(self.share_dir)]) == str(self.share_dir)
        assert os.path.commonpath([windows, str(self.share_dir)]) == str(self.share_dir)
        assert os.path.exists(os.path.join(data_dir, "TEST.parquet"))

    def test_rejects_inputs_outside_the_shared_directory(self, monkeypatch, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        captured = _run_capturing_job(monkeypatch, {"returncode": 0, "stdout": "", "stderr": ""})
        with pytest.raises(RuntimeError, match="共有ディレクトリの外"):
            run_sandboxed_evaluation(_HYPOTHESIS, str(outside), str(outside / "w.json"))
        assert "job" not in captured


class TestRunnerJob:
    @pytest.fixture(autouse=True)
    def _share(self, monkeypatch, tmp_path):
        self.share_dir = tmp_path / "share"
        self.share_dir.mkdir()
        monkeypatch.setattr(sandbox_executor, "FACTORY_SANDBOX_SHARE_DIR", str(self.share_dir))
        monkeypatch.setattr(sandbox_executor, "FACTORY_SANDBOX_SHARE_VOLUME", "vol_share")

    def test_sends_paths_relative_to_the_share(self, monkeypatch):
        data_dir, windows = prepare_sandbox_data({}, [])
        stdout = json.dumps({"status": "error", "traceback": "boom"}) + "\n"
        captured = _run_capturing_job(
            monkeypatch, {"returncode": 1, "stdout": stdout, "stderr": ""}
        )
        result = run_sandboxed_evaluation(_HYPOTHESIS, data_dir, windows)
        job = captured["job"]
        assert job["source_rel"].endswith("/candidate.py") and not job["source_rel"].startswith("/")
        assert job["data_rel"] == os.path.basename(data_dir)
        assert job["windows_rel"] == os.path.basename(windows)
        assert job["market"] == "us" and job["class_name"] == _HYPOTHESIS.rule_spec["class_name"]
        assert result.kind == "repairable"

    def test_ok_reply_is_evaluated(self, monkeypatch):
        data_dir, windows = prepare_sandbox_data({}, [])
        _run_capturing_job(monkeypatch, {"returncode": 0, "stdout": _ok_payload(), "stderr": ""})
        result = run_sandboxed_evaluation(_HYPOTHESIS, data_dir, windows)
        assert result.kind == "gate_evaluated"

    def test_runner_timeout_is_infra_error(self, monkeypatch):
        data_dir, windows = prepare_sandbox_data({}, [])
        _run_capturing_job(monkeypatch, {"timeout": True})
        result = run_sandboxed_evaluation(_HYPOTHESIS, data_dir, windows)
        assert result.kind == "infra_error" and "タイムアウト" in result.infra_detail

    def test_unreachable_runner_is_infra_error_not_repairable(self, monkeypatch):
        data_dir, windows = prepare_sandbox_data({}, [])

        def _down(job):
            raise RuntimeError("sandbox-runner に届きません: connection refused")

        monkeypatch.setattr(sandbox_executor, "_post_job", _down)
        result = run_sandboxed_evaluation(_HYPOTHESIS, data_dir, windows)
        assert result.kind == "infra_error" and "届きません" in result.infra_detail

    def test_input_missing_is_an_environment_failure_not_repairable(self, monkeypatch):
        data_dir, windows = prepare_sandbox_data({}, [])
        stdout = json.dumps({"status": "input_missing", "detail": "見つからない入力: ['/x']"})
        _run_capturing_job(monkeypatch, {"returncode": 0, "stdout": stdout, "stderr": ""})
        result = run_sandboxed_evaluation(_HYPOTHESIS, data_dir, windows)
        assert result.kind == "infra_error"
        assert result.environment_broken is True
        assert result.repair_detail is None


def test_post_job_requires_runner_url(monkeypatch):
    monkeypatch.setattr(sandbox_executor, "FACTORY_SANDBOX_RUNNER_URL", "")
    with pytest.raises(RuntimeError, match="FACTORY_SANDBOX_RUNNER_URL"):
        sandbox_executor._post_job({})


def test_share_is_required(monkeypatch):
    monkeypatch.setattr(sandbox_executor, "FACTORY_SANDBOX_SHARE_DIR", "")
    monkeypatch.setattr(sandbox_executor, "FACTORY_SANDBOX_SHARE_VOLUME", "")
    with pytest.raises(RuntimeError, match="共有ボリューム方式だけ"):
        _prepare()


@pytest.mark.parametrize(
    "share_dir,volume", [("/app/sandbox_share", ""), ("", "stockfixer_factory_sandbox")]
)
def test_half_configured_share_is_rejected(monkeypatch, share_dir, volume):
    """片方だけの設定で bind mount 方式に無言で落ちると #757 の全滅が再発するため例外にする。"""
    monkeypatch.setattr(sandbox_executor, "FACTORY_SANDBOX_SHARE_DIR", share_dir)
    monkeypatch.setattr(sandbox_executor, "FACTORY_SANDBOX_SHARE_VOLUME", volume)
    with pytest.raises(RuntimeError, match="両方"):
        _prepare()


class TestGeneratorStopsOnBrokenEnvironment:
    def test_stops_the_night_without_asking_claude_to_repair(self, monkeypatch):
        """入力が見えない失敗は候補を替えても直らないため、その晩の生成を打ち切る。"""
        monkeypatch.setattr(claude_rule_generator, "FACTORY_CLAUDE_RULEGEN_ENABLED", True)
        monkeypatch.setattr(claude_rule_generator, "FACTORY_CLAUDE_RULEGEN_COUNT", 3)
        generate = MagicMock(return_value=_HYPOTHESIS)
        monkeypatch.setattr(claude_rule_generator, "_generate_one_candidate", generate)
        broken = SandboxRunResult(kind="infra_error", infra_detail="x", environment_broken=True)
        with patch.object(
            claude_rule_generator, "run_sandboxed_evaluation", return_value=broken
        ) as run:
            result = claude_rule_generator.generate_claude_hypotheses(
                "us", 1.0, "/d", "/w.json", review_port=MagicMock()
            )

        assert result == []
        # 1 候補目の初回で打ち切る（修復依頼も 2・3 候補目の生成もしない）
        assert generate.call_count == 1
        assert run.call_count == 1

    def test_ordinary_infra_error_only_skips_the_candidate(self, monkeypatch):
        """タイムアウト等の一時的な失敗はその候補だけを諦め、次の候補へ進む（従来どおり）。"""
        monkeypatch.setattr(claude_rule_generator, "FACTORY_CLAUDE_RULEGEN_ENABLED", True)
        monkeypatch.setattr(claude_rule_generator, "FACTORY_CLAUDE_RULEGEN_COUNT", 3)
        generate = MagicMock(return_value=_HYPOTHESIS)
        monkeypatch.setattr(claude_rule_generator, "_generate_one_candidate", generate)
        timeout = SandboxRunResult(kind="infra_error", infra_detail="timeout")
        with patch.object(claude_rule_generator, "run_sandboxed_evaluation", return_value=timeout):
            claude_rule_generator.generate_claude_hypotheses(
                "us", 1.0, "/d", "/w.json", review_port=MagicMock()
            )

        assert generate.call_count == 3
