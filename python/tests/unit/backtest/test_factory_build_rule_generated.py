from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.factory import build_rule

_VALID_SOURCE = """
class GeneratedTestRule:
    name = "generated_test_rule"
    description = "test rule"

    def generate_signal(self, df):
        import pandas as pd
        return pd.Series(1, index=df.index)
"""


def test_generated_code_rejected_without_sandbox_flag(monkeypatch):
    monkeypatch.delenv("STOCKFIXER_SANDBOX", raising=False)
    spec = {
        "type": "generated_code",
        "source_code": _VALID_SOURCE,
        "class_name": "GeneratedTestRule",
        "rule_name": "generated_test_rule",
        "description": "test",
    }
    with pytest.raises(RuntimeError, match="サンドボックス"):
        build_rule(spec)


def test_generated_code_builds_rule_inside_sandbox(monkeypatch):
    monkeypatch.setenv("STOCKFIXER_SANDBOX", "1")
    spec = {
        "type": "generated_code",
        "source_code": _VALID_SOURCE,
        "class_name": "GeneratedTestRule",
        "rule_name": "generated_test_rule",
        "description": "test",
    }
    rule = build_rule(spec)
    assert rule.name == "generated_test_rule"
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
    signal = rule.generate_signal(df)
    assert list(signal) == [1, 1, 1]


def test_generated_code_missing_class_raises(monkeypatch):
    monkeypatch.setenv("STOCKFIXER_SANDBOX", "1")
    spec = {
        "type": "generated_code",
        "source_code": _VALID_SOURCE,
        "class_name": "NoSuchClass",
        "rule_name": "generated_test_rule",
        "description": "test",
    }
    with pytest.raises(ValueError, match="NoSuchClass"):
        build_rule(spec)
