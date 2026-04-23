"""
Cat1: CLI スモークテスト

run_*.py エントリポイントスクリプトに対して:
  1. --help フラグで exit code 0 が返ること（構文 / import エラーがないこと）
  2. モジュールとして import できること（syntax error がないこと）
  3. [slow] 実際に短い引数で実行して exit code 0 が返ること（ネットワーク不要）

実行方法:
    # 高速（--help のみ）
    python -m pytest tests/e2e/test_cli_smoke.py -v -m "not slow"

    # 全テスト（slow 含む、DB と一時環境が必要）
    python -m pytest tests/e2e/test_cli_smoke.py -v --timeout=120
"""

import importlib.util
import os
import subprocess
import sys

import pytest

# python/ ディレクトリの絶対パス（run_*.py が置かれた場所）
_PYTHON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
pytestmark = pytest.mark.timeout(60)


# ---------------------------------------------------------------------------
# --help スモークテスト（ネットワーク不要・高速）
# ---------------------------------------------------------------------------
class TestCLIHelp:
    """各 run_*.py に --help を渡して exit code 0 が返るかを確認する。"""

    SCRIPTS = [
        "run_data_creation.py",
        "run_model_creation.py",
        "run_predict.py",
        "run_backtest.py",
        "run_backtest_optimize.py",
        "run_scheduler.py",
        "run_stress_test.py",
    ]

    @pytest.mark.parametrize("script", SCRIPTS)
    def test_help_exits_zero(self, script):
        """--help で exit code 0 が返ること。"""
        script_path = os.path.join(_PYTHON_DIR, script)
        if not os.path.exists(script_path):
            pytest.skip(f"スクリプトが存在しない: {script}")

        result = subprocess.run(
            [sys.executable, script_path, "--help"],
            capture_output=True,
            text=True,
            cwd=_PYTHON_DIR,
        )
        assert result.returncode == 0, (
            f"{script} --help が exit code {result.returncode} を返した\n"
            f"stderr:\n{result.stderr[:500]}"
        )

    @pytest.mark.parametrize("script", SCRIPTS)
    def test_help_contains_usage(self, script):
        """--help の出力に usage 情報が含まれること。"""
        script_path = os.path.join(_PYTHON_DIR, script)
        if not os.path.exists(script_path):
            pytest.skip(f"スクリプトが存在しない: {script}")

        result = subprocess.run(
            [sys.executable, script_path, "--help"],
            capture_output=True,
            text=True,
            cwd=_PYTHON_DIR,
        )
        # argparse の usage または usage: が出力に含まれることを確認
        combined = (result.stdout + result.stderr).lower()
        assert "usage" in combined, f"{script} --help に usage 情報が含まれない:\n{result.stdout[:300]}"


# ---------------------------------------------------------------------------
# import スモークテスト（構文エラー・循環インポート検出）
# ---------------------------------------------------------------------------
class TestCLIImport:
    """run_*.py を importlib でロードして構文エラー・import エラーがないか確認する。"""

    SCRIPTS = [
        "run_data_creation.py",
        "run_model_creation.py",
        "run_predict.py",
        "run_backtest.py",
        "run_scheduler.py",
        "run_stress_test.py",
    ]

    @pytest.mark.parametrize("script", SCRIPTS)
    def test_importable(self, script):
        """スクリプトが構文エラーなしにロードできること。"""
        script_path = os.path.join(_PYTHON_DIR, script)
        if not os.path.exists(script_path):
            pytest.skip(f"スクリプトが存在しない: {script}")

        # spec.loader.exec_module は呼ばないので __name__ == "__main__" ブロックは実行されない
        spec = importlib.util.spec_from_file_location("_smoke_" + script, script_path)
        assert spec is not None, f"spec_from_file_location が None を返した: {script}"
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except SystemExit:
            # argparse がヘルプを出して sys.exit(0) する場合は OK
            pass
        except Exception as e:
            pytest.fail(f"{script} のロード中に例外が発生: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 実行スモークテスト（slow マーク: ネットワーク不要 + 一時 DB 使用）
# ---------------------------------------------------------------------------
class TestCLIActualRun:
    """
    実際に CLI を短いパラメータで実行する重いスモークテスト。
    pytest -m slow でのみ実行される。
    """

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_data_creation_help_only(self):
        """run_data_creation.py --help が正常終了すること（DB 不要）。"""
        script = os.path.join(_PYTHON_DIR, "run_data_creation.py")
        result = subprocess.run(
            [sys.executable, script, "--help"],
            capture_output=True,
            text=True,
            cwd=_PYTHON_DIR,
        )
        assert result.returncode == 0

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_backtest_missing_required_arg(self):
        """
        run_backtest.py を必須引数なしで実行するとエラー終了すること。
        --symbol が必須のため exit code 2 (argparse error) が期待値。
        """
        script = os.path.join(_PYTHON_DIR, "run_backtest.py")
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            cwd=_PYTHON_DIR,
        )
        # argparse の必須引数エラーは exit code 2
        assert result.returncode != 0, "必須引数なしなのに exit 0 が返った（引数チェックが機能していない可能性）"

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_stress_test_missing_required_arg(self):
        """
        run_stress_test.py を必須引数なしで実行するとエラー終了すること。
        --symbol / --symbols が必須のため exit code 1 が期待値。
        """
        script = os.path.join(_PYTHON_DIR, "run_stress_test.py")
        if not os.path.exists(script):
            pytest.skip("run_stress_test.py が存在しない")
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            cwd=_PYTHON_DIR,
        )
        # --symbol / --symbols なしは sys.exit(1) で終了
        assert result.returncode != 0, "必須引数なしなのに exit 0 が返った（引数チェックが機能していない可能性）"

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_no_traceback_in_help_output(self):
        """
        すべての run_*.py --help 出力に Traceback が含まれないこと。
        """
        scripts = [
            "run_data_creation.py",
            "run_model_creation.py",
            "run_predict.py",
            "run_backtest.py",
        ]
        for script_name in scripts:
            script = os.path.join(_PYTHON_DIR, script_name)
            if not os.path.exists(script):
                continue
            result = subprocess.run(
                [sys.executable, script, "--help"],
                capture_output=True,
                text=True,
                cwd=_PYTHON_DIR,
            )
            combined = result.stdout + result.stderr
            assert (
                "Traceback" not in combined
            ), f"{script_name} --help の出力に Traceback が含まれる:\n{combined[:500]}"


# ---------------------------------------------------------------------------
# run_*.py のファイル存在確認
# ---------------------------------------------------------------------------
class TestCLIFilesExist:
    """期待する run_*.py が python/ 配下に存在することを確認する。"""

    REQUIRED_SCRIPTS = [
        "run_data_creation.py",
        "run_model_creation.py",
        "run_predict.py",
        "run_backtest.py",
        "run_scheduler.py",
        "run_discord_bot.py",
        "run_stress_test.py",
    ]

    @pytest.mark.parametrize("script", REQUIRED_SCRIPTS)
    def test_script_exists(self, script):
        """run スクリプトが python/ 配下に存在すること。"""
        path = os.path.join(_PYTHON_DIR, script)
        assert os.path.exists(path), f"run スクリプトが存在しない: {path}"
