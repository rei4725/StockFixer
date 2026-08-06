from __future__ import annotations

from config.settings import (
    FACTORY_GATE_MAX_DRAWDOWN,
    FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS,
    FACTORY_GATE_MIN_TRADES_PER_SYMBOL,
)
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


def test_issue_body_reports_symbol_denominators(tmp_path, monkeypatch):
    """Sharpe の母数が読み取れることを保証する（#625）。

    従来の「銘柄数 194」はデータ取得できた銘柄数であり Sharpe の母数ではなく、
    レビュー時に誤読を招いていた。
    """
    monkeypatch.setattr("src.backtest.factory_report.get_results_dir", lambda: str(tmp_path))

    hypothesis = FactoryHypothesis(
        rule_spec={"type": "atomic", "rule": "rsi_contrarian", "params": {}},
        market="jp",
    )
    evaluation = FactoryEvaluation(
        hypothesis=hypothesis,
        sharpe_ratio=1.6,
        dsr=0.99,
        pbo=0.1,
        num_trades=85,
        max_drawdown=-0.19,
        win_rate=0.85,
        total_return=0.09,
        window_returns=[0.01, 0.02],
        n_symbols=194,
        n_symbols_with_signal=69,
        n_effective_symbols=16,
        avg_trades_per_symbol=1.23,
    )

    path = write_report(evaluation, champion_sharpe=1.083, period=("2024-07-25", "2026-07-25"))

    import json

    with open(path, encoding="utf-8") as f:
        report = json.load(f)

    body = report["issue_body"]
    assert "データ取得銘柄数 194" in body
    assert "シグナル発生銘柄" in body
    assert "| 69 |" in body
    # 有効銘柄の行はラベル・値・ゲート列をまとめてピン留めする。
    # n_effective_symbols が別フィールド（例: n_symbols_with_signal）に
    # 誤配線されていても、"有効銘柄" という部分文字列だけでは
    # 「Sharpe（有効銘柄平均）」等の他の行にも一致してしまい検出できないため。
    effective_symbols_row = (
        f"| 有効銘柄（{FACTORY_GATE_MIN_TRADES_PER_SYMBOL}取引以上） "
        f"| 16 | >= {FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS} |"
    )
    assert effective_symbols_row in body
    assert "銘柄あたり平均取引数（シグナル発生銘柄基準）" in body
    assert "1.23" in body
    # 母数が曖昧だった旧ラベルは残っていない
    assert "Sharpe（銘柄平均）" not in body
    assert "Sharpe（有効銘柄平均）" in body
    # 最大DD は有効銘柄の中での最悪値であることをラベル・値ごとピン留めする（#625 Finding 1）。
    max_drawdown_row = (
        f"| 最大DD（有効銘柄の最悪値） | -19.00% | >= {FACTORY_GATE_MAX_DRAWDOWN:.0%} |"
    )
    assert max_drawdown_row in body

    assert report["gate"]["n_symbols_with_signal"] == 69
    assert report["gate"]["n_effective_symbols"] == 16
    assert report["gate"]["avg_trades_per_symbol"] == 1.23
