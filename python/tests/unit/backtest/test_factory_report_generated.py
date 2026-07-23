from __future__ import annotations

from src.backtest.factory_report import write_report
from src.backtest.types import FactoryEvaluation, FactoryHypothesis


def test_issue_body_renders_source_code_block(tmp_path, monkeypatch):
    monkeypatch.setattr("src.backtest.factory_report.get_results_dir", lambda: str(tmp_path))

    hypothesis = FactoryHypothesis(
        rule_spec={
            "type": "generated_code",
            "source_code": "class NovelRule:\n    name = 'novel_rule'\n",
            "class_name": "NovelRule",
            "rule_name": "novel_rule",
            "description": "新しい着眼点のルール",
        },
        market="us",
    )
    evaluation = FactoryEvaluation(
        hypothesis=hypothesis,
        sharpe_ratio=1.8,
        dsr=0.96,
        pbo=0.2,
        num_trades=45,
        max_drawdown=-0.1,
        win_rate=0.55,
        total_return=0.2,
        window_returns=[0.01, 0.02],
        n_symbols=3,
    )
    path = write_report(evaluation, champion_sharpe=1.0, period=("2024-01-01", "2025-01-01"))

    import json

    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    body = report["issue_body"]
    assert "```python" in body
    assert "class NovelRule" in body
    assert "新しい着眼点のルール" in body
