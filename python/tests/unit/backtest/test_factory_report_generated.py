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
        portfolio_sharpe_ratio=0.94,
        dsr=0.99,
        pbo=0.1,
        num_trades=85,
        max_drawdown=-0.19,
        portfolio_max_drawdown=-0.07,
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
    # Sharpe も2行。champion 列が付くのはプール済み per-trade ベースの値。
    assert "| Sharpe（プール済み取引リターンを年率化） | 0.940 |" in body
    assert "| Sharpe（有効銘柄平均・診断用） | 1.600 | - |" in body
    # DD は2行に分かれる。ゲート列が付くのはポートフォリオDD、最悪銘柄DDは診断用。
    # どちらの数字がゲート判定に使われたかをレビュー時に取り違えないようピン留めする。
    portfolio_dd_row = (
        f"| 最大DD（有効銘柄を等金額保有したポートフォリオ） "
        f"| -7.00% | >= {FACTORY_GATE_MAX_DRAWDOWN:.0%} |"
    )
    assert portfolio_dd_row in body
    assert "| 最大DD（有効銘柄の最悪値・診断用） | -19.00% | - |" in body

    assert report["gate"]["n_symbols_with_signal"] == 69
    assert report["gate"]["n_effective_symbols"] == 16
    assert report["gate"]["avg_trades_per_symbol"] == 1.23
    assert report["gate"]["portfolio_max_drawdown"] == -0.07
    assert report["gate"]["portfolio_sharpe_ratio"] == 0.94


def test_issue_body_marks_worst_symbol_dd_as_gate_input_when_portfolio_dd_missing(
    tmp_path, monkeypatch
):
    """equity 曲線が取れずフォールバックしたことがレポートから判別できること。

    ポートフォリオDDが算出できなかった夜に、最悪銘柄DDで判定されたことを
    読み手が気づけないと、ゲートが無言で従来挙動に戻っていても分からない。
    """
    monkeypatch.setattr("src.backtest.factory_report.get_results_dir", lambda: str(tmp_path))

    evaluation = FactoryEvaluation(
        hypothesis=FactoryHypothesis(
            rule_spec={"type": "atomic", "rule": "rsi_contrarian", "params": {}}, market="jp"
        ),
        sharpe_ratio=1.6,
        dsr=0.99,
        pbo=0.1,
        num_trades=85,
        max_drawdown=-0.19,
        win_rate=0.85,
        total_return=0.09,
        window_returns=[0.01],
        n_symbols=20,
    )

    path = write_report(evaluation, champion_sharpe=1.0, period=("2024-01-01", "2025-01-01"))

    import json

    with open(path, encoding="utf-8") as f:
        report = json.load(f)

    body = report["issue_body"]
    assert "プール値算出不能によりゲート判定に使用" in body
    assert "曲線欠損によりゲート判定に使用" in body
    assert "等金額保有したポートフォリオ" not in body
    # NaN は JSON に書かず null にする
    assert report["gate"]["portfolio_max_drawdown"] is None
    assert report["gate"]["portfolio_sharpe_ratio"] is None
