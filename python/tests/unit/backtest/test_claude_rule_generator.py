import json
from unittest.mock import MagicMock, patch

from src.backtest.claude_rule_generator import generate_claude_hypotheses
from src.backtest.sandbox_executor import SandboxRunResult
from src.backtest.types import FactoryEvaluation, FactoryHypothesis

_VALID_RESPONSE = json.dumps(
    {
        "rule_name": "novel_rule",
        "class_name": "NovelRule",
        "description": "何か新しいルール",
        "source_code": "class NovelRule:\n    name = 'novel_rule'\n"
        "    description = 'x'\n    def generate_signal(self, df):\n"
        "        import pandas as pd\n        return pd.Series(0, index=df.index)\n",
    }
)


def _make_hypothesis() -> FactoryHypothesis:
    return FactoryHypothesis(
        rule_spec={
            "type": "generated_code",
            "source_code": "class NovelRule:\n    pass\n",
            "class_name": "NovelRule",
            "rule_name": "novel_rule",
            "description": "x",
        },
        market="us",
    )


def test_disabled_flag_returns_empty(monkeypatch):
    monkeypatch.setattr("src.backtest.claude_rule_generator.FACTORY_CLAUDE_RULEGEN_ENABLED", False)
    result = generate_claude_hypotheses(
        market="us",
        champion_sharpe=1.0,
        shared_data_dir="dummy",
        windows_file="dummy.json",
    )
    assert result == []


@patch("src.backtest.claude_rule_generator.run_sandboxed_evaluation")
@patch("src.backtest.claude_rule_generator.get_text_review_port")
def test_gate_evaluated_candidate_returned(mock_port_factory, mock_sandbox, monkeypatch):
    monkeypatch.setattr("src.backtest.claude_rule_generator.FACTORY_CLAUDE_RULEGEN_ENABLED", True)
    monkeypatch.setattr("src.backtest.claude_rule_generator.FACTORY_CLAUDE_RULEGEN_COUNT", 1)

    mock_port = MagicMock()
    mock_port.complete.return_value = _VALID_RESPONSE
    mock_port_factory.return_value = mock_port

    evaluation = FactoryEvaluation(hypothesis=_make_hypothesis(), sharpe_ratio=1.5, num_trades=50)
    mock_sandbox.return_value = SandboxRunResult(kind="gate_evaluated", evaluation=evaluation)

    result = generate_claude_hypotheses(
        market="us",
        champion_sharpe=1.0,
        shared_data_dir="dummy",
        windows_file="dummy.json",
    )
    assert len(result) == 1
    assert result[0].sharpe_ratio == 1.5
    mock_port.complete.assert_called_once()
    mock_sandbox.assert_called_once()


@patch("src.backtest.claude_rule_generator.run_sandboxed_evaluation")
@patch("src.backtest.claude_rule_generator.get_text_review_port")
def test_repairable_failure_retries_then_gives_up(mock_port_factory, mock_sandbox, monkeypatch):
    monkeypatch.setattr("src.backtest.claude_rule_generator.FACTORY_CLAUDE_RULEGEN_ENABLED", True)
    monkeypatch.setattr("src.backtest.claude_rule_generator.FACTORY_CLAUDE_RULEGEN_COUNT", 1)
    monkeypatch.setattr(
        "src.backtest.claude_rule_generator.FACTORY_CLAUDE_RULEGEN_MAX_REPAIR_ATTEMPTS", 2
    )

    mock_port = MagicMock()
    mock_port.complete.return_value = _VALID_RESPONSE
    mock_port_factory.return_value = mock_port

    mock_sandbox.return_value = SandboxRunResult(
        kind="repairable", repair_detail="静的検査で拒否: import os"
    )

    result = generate_claude_hypotheses(
        market="us",
        champion_sharpe=1.0,
        shared_data_dir="dummy",
        windows_file="dummy.json",
    )
    assert result == []
    # 初回 + 修復2回 = 3回呼ばれる
    assert mock_port.complete.call_count == 3
    assert mock_sandbox.call_count == 3


@patch("src.backtest.claude_rule_generator.run_sandboxed_evaluation")
@patch("src.backtest.claude_rule_generator.get_text_review_port")
def test_infra_error_does_not_consume_repair_budget(mock_port_factory, mock_sandbox, monkeypatch):
    monkeypatch.setattr("src.backtest.claude_rule_generator.FACTORY_CLAUDE_RULEGEN_ENABLED", True)
    monkeypatch.setattr("src.backtest.claude_rule_generator.FACTORY_CLAUDE_RULEGEN_COUNT", 1)

    mock_port = MagicMock()
    mock_port.complete.return_value = _VALID_RESPONSE
    mock_port_factory.return_value = mock_port

    mock_sandbox.return_value = SandboxRunResult(kind="infra_error", infra_detail="timeout")

    result = generate_claude_hypotheses(
        market="us",
        champion_sharpe=1.0,
        shared_data_dir="dummy",
        windows_file="dummy.json",
    )
    assert result == []
    # インフラ起因は修復リトライしない（1回のみ呼ばれる）
    assert mock_port.complete.call_count == 1
    assert mock_sandbox.call_count == 1
