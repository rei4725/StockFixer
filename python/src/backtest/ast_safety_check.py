"""Claude生成ルールコードの静的安全性検査。

execする前に構文木を走査し、危険なimport/呼び出しを拒否する。これは早期に
明らかな不正コードを弾く高速フィルタであり、主たる安全境界ではない
（Dockerサンドボックスの --network none / 読み取り専用マウントが主たる境界）。
迂回可能であることを前提に、過信しないこと。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

_BANNED_IMPORTS = frozenset(
    {
        "os",
        "subprocess",
        "socket",
        "sys",
        "shutil",
        "pathlib",
        "importlib",
        "ctypes",
        "multiprocessing",
        "threading",
        "urllib",
        "requests",
        "http",
        "ftplib",
        "smtplib",
    }
)
_BANNED_CALLS = frozenset({"eval", "exec", "compile", "open", "__import__", "input"})
_BANNED_DUNDER_ATTRS = frozenset(
    {"__subclasses__", "__globals__", "__builtins__", "__import__", "__loader__"}
)


@dataclass
class SafetyViolation:
    line: int
    reason: str


@dataclass
class SafetyCheckResult:
    passed: bool
    violations: list[SafetyViolation] = field(default_factory=list)


def check_source_safety(source_code: str) -> SafetyCheckResult:
    """生成コードの構文木を走査し、危険な構文を検出する。

    構文エラー自体も違反として扱う。
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return SafetyCheckResult(
            passed=False,
            violations=[SafetyViolation(line=exc.lineno or 1, reason=f"構文エラー: {exc.msg}")],
        )

    violations: list[SafetyViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _BANNED_IMPORTS:
                    violations.append(
                        SafetyViolation(line=node.lineno, reason=f"禁止importです: {alias.name}")
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _BANNED_IMPORTS:
                violations.append(
                    SafetyViolation(line=node.lineno, reason=f"禁止importです: {node.module}")
                )
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BANNED_CALLS:
                violations.append(
                    SafetyViolation(line=node.lineno, reason=f"禁止呼び出しです: {func.id}(...)")
                )
        elif isinstance(node, ast.Attribute):
            if node.attr in _BANNED_DUNDER_ATTRS:
                violations.append(
                    SafetyViolation(line=node.lineno, reason=f"禁止属性アクセスです: .{node.attr}")
                )

    return SafetyCheckResult(passed=not violations, violations=violations)
