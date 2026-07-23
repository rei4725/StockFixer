from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from src.backtest.factory import run_factory_batch
from src.backtest.types import FactoryEvaluation, FactoryHypothesis


def _sample_data():
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    return pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1_000_000},
        index=dates,
    )


@patch("src.backtest.factory.prepare_sandbox_data")
@patch("src.backtest.factory.generate_claude_hypotheses")
@patch("src.backtest.factory._load_symbol_data")
def test_claude_hypotheses_included_when_enabled(
    mock_load_data, mock_generate, mock_prepare, monkeypatch
):
    monkeypatch.setattr("src.backtest.factory.FACTORY_CLAUDE_RULEGEN_ENABLED", True)
    mock_load_data.return_value = {"TEST": _sample_data()}
    mock_prepare.return_value = ("/tmp/dummy_data", "/tmp/dummy_windows.json")

    claude_hypothesis = FactoryHypothesis(
        rule_spec={
            "type": "generated_code",
            "source_code": "class X:\n    pass\n",
            "class_name": "X",
            "rule_name": "claude_rule",
            "description": "x",
        },
        market="us",
    )
    mock_generate.return_value = [
        FactoryEvaluation(
            hypothesis=claude_hypothesis,
            sharpe_ratio=2.0,
            sharpe_per_trade=0.1,
            num_trades=40,
            max_drawdown=-0.05,
            # 実装では sandbox_executor が windows_file の窓数ぶん window_returns を
            # 埋めて返す（sandbox_executor.py の FactoryEvaluation 構築を参照）。
            # ここでは run_factory_batch() 側の PBO 行列計算（窓数で揃った
            # 2次元配列を要求する）と整合させるため、n_windows=4 と同じ長さで
            # window_returns を明示する。
            window_returns=[0.02, 0.01, 0.03, 0.01],
        )
    ]

    result = run_factory_batch(market="us", symbols=["TEST"], budget=1, n_windows=4)

    labels = [e.hypothesis.rule_spec.get("rule_name") for e in result.evaluated]
    assert "claude_rule" in labels
    mock_generate.assert_called_once()


@patch("src.backtest.factory.generate_claude_hypotheses")
@patch("src.backtest.factory._load_symbol_data")
def test_claude_hypotheses_skipped_when_disabled(mock_load_data, mock_generate, monkeypatch):
    monkeypatch.setattr("src.backtest.factory.FACTORY_CLAUDE_RULEGEN_ENABLED", False)
    mock_load_data.return_value = {"TEST": _sample_data()}

    run_factory_batch(market="us", symbols=["TEST"], budget=1, n_windows=4)

    mock_generate.assert_not_called()
