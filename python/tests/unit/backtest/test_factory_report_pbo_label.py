"""factory レポートの PBO 行がゲート条件に見えないこと（#738）。

PBO はバッチ単位の診断値で apply_gate の判定に使っていない。表のゲート列に
「<= 0.5」と書くと、PBO 超過なのに合格している矛盾に見える。
"""

from src.backtest.factory_report import _build_issue_body
from src.backtest.types import FactoryEvaluation, FactoryHypothesis

_SPEC = {"type": "atomic", "rule": "rsi_contrarian", "params": {}}


def _body(pbo: float) -> str:
    evaluation = FactoryEvaluation(
        hypothesis=FactoryHypothesis(rule_spec=_SPEC, market="jp"),
        sharpe_ratio=1.5,
        dsr=0.96,
        pbo=pbo,
        num_trades=40,
        max_drawdown=-0.1,
        window_returns=[0.01] * 8,
        n_symbols=3,
    )
    return _build_issue_body(evaluation, champion_sharpe=1.0, period=("2024-01-01", "2026-01-01"))


def test_pbo_row_is_labelled_as_batch_diagnostic():
    body = _body(pbo=0.543)
    pbo_rows = [line for line in body.splitlines() if line.startswith("| PBO")]

    assert pbo_rows == ["| PBO（バッチ全体の診断値） | 0.543 | - （ゲート判定には不使用） |"]


def test_high_pbo_warning_still_present():
    assert "バッチPBO=0.543" in _body(pbo=0.543)
