from src.backtest.ast_safety_check import check_source_safety


def test_safe_source_passes():
    source = """
import pandas as pd


class SafeRule:
    name = "safe_rule"
    description = "test"

    def generate_signal(self, df):
        return (df["Close"] > df["Close"].rolling(20).mean()).astype(int)
"""
    result = check_source_safety(source)
    assert result.passed is True
    assert result.violations == []


def test_banned_import_os_rejected():
    source = "import os\n\nclass R:\n    pass\n"
    result = check_source_safety(source)
    assert result.passed is False
    assert any("os" in v.reason for v in result.violations)
    assert result.violations[0].line == 1


def test_banned_import_from_subprocess_rejected():
    source = "from subprocess import run\n\nclass R:\n    pass\n"
    result = check_source_safety(source)
    assert result.passed is False
    assert any("subprocess" in v.reason for v in result.violations)


def test_banned_call_open_rejected():
    source = "class R:\n    def f(self):\n        open('/etc/passwd')\n"
    result = check_source_safety(source)
    assert result.passed is False
    assert any("open" in v.reason for v in result.violations)


def test_banned_call_eval_rejected():
    source = "class R:\n    def f(self):\n        eval('1+1')\n"
    result = check_source_safety(source)
    assert result.passed is False
    assert any("eval" in v.reason for v in result.violations)


def test_banned_dunder_subclasses_rejected():
    source = "class R:\n    def f(self):\n        object.__subclasses__()\n"
    result = check_source_safety(source)
    assert result.passed is False
    assert any("__subclasses__" in v.reason for v in result.violations)


def test_syntax_error_rejected():
    source = "def broken(:\n"
    result = check_source_safety(source)
    assert result.passed is False
    assert "構文エラー" in result.violations[0].reason
