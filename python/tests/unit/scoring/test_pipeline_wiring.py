"""採点の段が日次・週次の道につながっていることを、ソースの構文木で確かめる。

tests/unit/conftest.py の autouse の差し替え（_stub_scoring_stages）は実行時の
呼び出しを空にするので、ここではコードを動かさずに構文木だけを読む。
このファイルは錨の一部（自動マージの禁止リストにある）。
"""

from __future__ import annotations

import ast
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[3]
JOBS = PYTHON_ROOT / "src" / "orchestration" / "jobs"
STAGE_MODULE = "src.orchestration.jobs.scoring_stage"


def _tree(name: str) -> ast.Module:
    path = JOBS / name
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(found) == 1, f"{name} がモジュールの直下に1つだけあること"
    return found[0]


def _calls(func: ast.FunctionDef, name: str) -> list[ast.Call]:
    """func の中で name(...) と呼んでいる箇所（入れ子の関数の中は数えない）。"""
    out: list[ast.Call] = []
    stack: list[ast.AST] = list(func.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == name:
                out.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return sorted(out, key=lambda c: (c.lineno, c.col_offset))


def _first_line(func: ast.FunctionDef, name: str) -> int:
    calls = _calls(func, name)
    assert calls, f"{func.name} が {name} を呼ぶこと"
    return calls[0].lineno


def _imported_from(tree: ast.Module, module: str) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.update(a.name for a in node.names if a.asname is None)
    return names


def _defined_at_top(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def test_daily_pipeline_calls_scoring_stages_in_order() -> None:
    tree = _tree("daily.py")
    stages = {"snapshot_predictions_stage", "daily_scoring_stage", "log_rule_signal_versions_stage"}
    assert stages <= _imported_from(tree, STAGE_MODULE)
    assert not stages & _defined_at_top(tree), "daily.py で段を定義し直さないこと"

    run = _function(tree, "run_daily_pipeline")
    shadow = _first_line(run, "run_challenger_shadow_stage")
    snapshot = _first_line(run, "snapshot_predictions_stage")
    sectors = _first_line(run, "append_accuracy_json_sectors")
    scoring = _first_line(run, "daily_scoring_stage")
    assert shadow < snapshot, "予想の記録は challenger の影の予想の後"
    assert sectors < scoring, "ルールシグナルの採点は業種の内訳の書き足しの後"


def test_daily_rule_signals_logs_versions() -> None:
    run = _function(_tree("daily.py"), "run_daily_rule_signals")
    assert _calls(run, "log_rule_signal_versions_stage")


def test_weekly_rule_evaluation_calls_scoring_stage() -> None:
    tree = _tree("weekly.py")
    assert "weekly_rule_scoring_stage" in _imported_from(tree, STAGE_MODULE)
    assert "weekly_rule_scoring_stage" not in _defined_at_top(tree)
    run = _function(tree, "run_weekly_rule_evaluation")
    assert _calls(run, "weekly_rule_scoring_stage")


def test_daily_scoring_stage_uses_defaults() -> None:
    tree = _tree("scoring_stage.py")
    assert "run_daily_scoring" not in _defined_at_top(tree)
    stage = _function(tree, "daily_scoring_stage")
    imports = [
        n
        for n in ast.walk(stage)
        if isinstance(n, ast.ImportFrom) and n.module == "src.scoring.rule_report"
    ]
    assert any(a.name == "run_daily_scoring" and a.asname is None for n in imports for a in n.names)
    calls = _calls(stage, "run_daily_scoring")
    assert len(calls) == 1
    assert not calls[0].args and not calls[0].keywords, "market と as_of は既定（jp・今日）のまま"
