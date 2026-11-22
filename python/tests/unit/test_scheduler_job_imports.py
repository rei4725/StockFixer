"""SCHEDULE_CONFIG の各ジョブの関数が中で読む import の名前が解けることの見張り（#2092）

ジョブの関数は本体の中で遅れて import するため、re-export を消したり関数を移したりすると、
その定時に初めて ImportError で落ちる。ここでは run_scheduler.py を ast で読み、
各ジョブの関数の from X import Y を importlib.import_module と hasattr で解く。
ジョブの関数そのものは呼ばない。
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import run_scheduler

_SOURCE_PATH = Path(run_scheduler.__file__)


def _job_functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("job_")
    }


def _config_func_names(tree: ast.Module) -> dict[str, str]:
    """SCHEDULE_CONFIG の各ジョブの名前と、その func に書かれた名前を返す。"""
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "SCHEDULE_CONFIG" for t in node.targets)
            and isinstance(node.value, ast.Dict)
        ):
            result: dict[str, str] = {}
            for key, value in zip(node.value.keys, node.value.values):
                assert isinstance(key, ast.Constant) and isinstance(value, ast.Dict)
                func_name = ""
                for k, v in zip(value.keys, value.values):
                    if isinstance(k, ast.Constant) and k.value == "func":
                        func_name = v.id if isinstance(v, ast.Name) else ast.dump(v)
                result[str(key.value)] = func_name
            return result
    raise AssertionError("SCHEDULE_CONFIG が見つからない")


def _unresolved_imports(tree: ast.Module) -> tuple[list[str], int]:
    """解けない import と、見たジョブの本数を返す。"""
    jobs = _job_functions(tree)
    bad: list[str] = []
    checked = 0
    for job_name, func_name in _config_func_names(tree).items():
        func = jobs.get(func_name)
        if func is None:
            bad.append(f"{job_name}: func {func_name!r} は run_scheduler の job_ の関数ではない")
            continue
        checked += 1
        for node in ast.walk(func):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level or not node.module:
                bad.append(f"{job_name}: 相対の import は解かない")
                continue
            try:
                module = importlib.import_module(node.module)
            except ImportError as exc:
                bad.append(f"{job_name}: {node.module} を読めない ({exc})")
                continue
            for alias in node.names:
                if not hasattr(module, alias.name):
                    bad.append(f"{job_name}: {node.module}.{alias.name} が無い")
    return bad, checked


def _parse(source: str) -> ast.Module:
    return ast.parse(source, filename=str(_SOURCE_PATH))


def test_every_job_import_resolves() -> None:
    """今の SCHEDULE_CONFIG の全ジョブで、関数の中の import が全部解ける。"""
    tree = _parse(_SOURCE_PATH.read_text(encoding="utf-8"))
    bad, checked = _unresolved_imports(tree)
    assert bad == []
    assert checked == len(run_scheduler.SCHEDULE_CONFIG)


def test_ast_func_names_match_runtime_config() -> None:
    """ast で読んだ func が、実際の SCHEDULE_CONFIG の func と同じ run_scheduler の関数である。"""
    tree = _parse(_SOURCE_PATH.read_text(encoding="utf-8"))
    names = _config_func_names(tree)
    assert set(names) == set(run_scheduler.SCHEDULE_CONFIG)
    for job_name, job in run_scheduler.SCHEDULE_CONFIG.items():
        func = job["func"]
        assert getattr(func, "__module__", None) == run_scheduler.__name__
        assert getattr(func, "__name__", None) == names[job_name]


def test_guard_catches_missing_import_name() -> None:
    """見張りが空振りしないこと: import の名前を1つ壊すと当たる。"""
    tree = _parse(_SOURCE_PATH.read_text(encoding="utf-8"))
    first_job, first_func = next(iter(_config_func_names(tree).items()))
    func = _job_functions(tree)[first_func]
    node = next(n for n in ast.walk(func) if isinstance(n, ast.ImportFrom))
    node.names[0].name = "no_such_name_for_guard_test"
    bad, _ = _unresolved_imports(tree)
    assert bad == [f"{first_job}: {node.module}.no_such_name_for_guard_test が無い"]
