from __future__ import annotations

import os
import shutil
import subprocess

import pandas as pd
import pytest

from src.backtest.sandbox_executor import prepare_sandbox_data, run_sandboxed_evaluation
from src.backtest.types import FactoryHypothesis


def _sandbox_image_available() -> bool:
    if shutil.which("docker") is None:
        return False
    image = os.environ.get("FACTORY_SANDBOX_IMAGE", "stockfixer:dev")
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not _sandbox_image_available(),
    reason="Docker またはサンドボックス用イメージ（stockfixer:dev 等）が利用できない環境ではスキップ",
)


def _sample_data() -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    df = pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1_000_000},
        index=dates,
    )
    return {"TEST": df}


def _windows():
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    return [(dates[0], dates[29]), (dates[29], dates[-1])]


def test_safe_rule_evaluates_successfully():
    data_dir, windows_file = prepare_sandbox_data(_sample_data(), _windows())
    hypothesis = FactoryHypothesis(
        rule_spec={
            "type": "generated_code",
            "source_code": (
                "class SafeRule:\n"
                "    name = 'safe_rule'\n"
                "    description = 'test'\n"
                "    def generate_signal(self, df):\n"
                "        import pandas as pd\n"
                "        return pd.Series(0, index=df.index)\n"
            ),
            "class_name": "SafeRule",
            "rule_name": "safe_rule",
            "description": "test",
        },
        market="us",
    )
    result = run_sandboxed_evaluation(hypothesis, data_dir, windows_file)
    assert result.kind == "gate_evaluated"
    assert result.evaluation is not None
    assert result.evaluation.n_symbols == 1


def test_banned_import_rejected_without_docker_run():
    data_dir, windows_file = prepare_sandbox_data(_sample_data(), _windows())
    hypothesis = FactoryHypothesis(
        rule_spec={
            "type": "generated_code",
            "source_code": "import os\n\nclass BadRule:\n    pass\n",
            "class_name": "BadRule",
            "rule_name": "bad_rule",
            "description": "test",
        },
        market="us",
    )
    result = run_sandboxed_evaluation(hypothesis, data_dir, windows_file)
    assert result.kind == "repairable"
    assert "os" in result.repair_detail


def test_crashing_rule_returns_repairable_with_traceback():
    data_dir, windows_file = prepare_sandbox_data(_sample_data(), _windows())
    hypothesis = FactoryHypothesis(
        rule_spec={
            "type": "generated_code",
            "source_code": (
                "class CrashingRule:\n"
                "    name = 'crashing_rule'\n"
                "    description = 'test'\n"
                "    def generate_signal(self, df):\n"
                "        raise ValueError('boom')\n"
            ),
            "class_name": "CrashingRule",
            "rule_name": "crashing_rule",
            "description": "test",
        },
        market="us",
    )
    result = run_sandboxed_evaluation(hypothesis, data_dir, windows_file)
    assert result.kind == "repairable"
    assert "boom" in result.repair_detail
