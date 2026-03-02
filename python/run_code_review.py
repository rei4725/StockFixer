#!/usr/bin/env python3
"""
コードレビュー実行スクリプト

使用方法:
    python python/run_code_review.py                    # 全ファイルをレビュー
    python python/run_code_review.py python/src/services/data_pipeline.py  # 特定ファイルのみ
    python python/run_code_review.py --full            # テスト+カバレッジを含む完全レビュー
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional


class CodeReviewRunner:
    """コードレビュー実行エンジン"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}

        # プロジェクトルートを特定（スクリプトの親の親）
        self.project_root = Path(__file__).parent.parent
        self.venv_python = self.project_root / ".venv/Scripts/python.exe"

        # Python実行コマンドを設定
        if self.venv_python.exists():
            self.python_cmd = str(self.venv_python)
        else:
            self.python_cmd = "python"

    def run_command(self, cmd: List[str], name: str) -> bool:
        """コマンドを実行してパス/失敗を記録"""
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🔍 {name}")
            print(f"{'='*70}")
            print(f"$ {' '.join(cmd)}\n")

        try:
            result = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                check=False,
                cwd=str(self.project_root),
            )
            passed = result.returncode == 0
            self.results[name] = passed
            return passed
        except Exception as e:
            print(f"❌ エラー実行: {e}")
            self.results[name] = False
            return False

    def run_code_quality_checks(self, files: Optional[List[str]] = None):
        """コード品質チェック実行"""
        print("\n" + "=" * 70)
        print("📋 コード品質チェック開始")
        print("=" * 70)

        target = files if files else ["python/src", "python/run_*.py"]
        file_args = [f for f in target if f]

        # Black チェック
        self.run_command(
            [self.python_cmd, "-m", "black", "--check", "--line-length=100"] + file_args,
            "Black コード整形チェック",
        )

        # isort チェック
        self.run_command(
            [self.python_cmd, "-m", "isort", "--check", "--profile=black", "--line-length=100"]
            + file_args,
            "isort インポート整理チェック",
        )

        # Flake8 実行
        self.run_command(
            [self.python_cmd, "-m", "flake8"]
            + file_args
            + ["--max-line-length=100", "--exclude=__pycache__,.venv,.git"],
            "Flake8 PEP8 チェック",
        )

        # mypy 実行
        self.run_command(
            [self.python_cmd, "-m", "mypy"]
            + ["python/src"]
            + ["--ignore-missing-imports", "--skip-validation"],
            "mypy 型安全性チェック",
        )

    def run_lock_detection(self, files: Optional[List[str]] = None):
        """ロック問題検出実行"""
        print("\n" + "=" * 70)
        print("🔒 ロック問題検出開始")
        print("=" * 70)

        target = files if files else ["python/src"]

        # DuckDB並行処理チェック
        self.run_command(
            [self.python_cmd, "./.github/hooks/check_duckdb_concurrency.py"] + target,
            "DuckDB並行処理チェック",
        )

        # ファイルロック検出
        self.run_command(
            [self.python_cmd, "./.github/hooks/check_file_lock.py"] + target,
            "ファイルロック検出",
        )

    def run_tests(self, full: bool = False):
        """テスト実行"""
        print("\n" + "=" * 70)
        print("✅ テスト実行開始")
        print("=" * 70)

        # ユニットテスト
        self.run_command(
            [self.python_cmd, "-m", "pytest", "tests/unit/", "-v", "--tb=short"],
            "ユニットテスト実行",
        )

        if full:
            # 統合テスト
            self.run_command(
                [self.python_cmd, "-m", "pytest", "tests/integration/", "-v", "--tb=short"],
                "統合テスト実行",
            )

            # カバレッジ
            self.run_command(
                [
                    self.python_cmd,
                    "-m",
                    "pytest",
                    "tests/",
                    "--cov=python/src",
                    "--cov-report=term-missing",
                    "--cov-report=html",
                ],
                "カバレッジレポート(HTML生成)",
            )

    def print_summary(self):
        """結果サマリーを表示"""
        print("\n" + "=" * 70)
        print("📊 コードレビュー結果サマリー")
        print("=" * 70)

        passed_count = sum(1 for v in self.results.values() if v)
        total_count = len(self.results)

        for check_name, result in self.results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status:10} {check_name}")

        print(f"\n合計: {passed_count}/{total_count} パス")

        if passed_count == total_count:
            print("\n🎉 すべてのチェックがパスしました！")
            return True
        else:
            print(f"\n⚠️  {total_count - passed_count} 個のチェックが失敗しました。")
            return False

    def run_auto_fix(self):
        """自動修正を実行"""
        print("\n" + "=" * 70)
        print("🔧 自動修正開始")
        print("=" * 70)

        # Black 自動修正
        self.run_command(
            [self.python_cmd, "-m", "black", "--line-length=100", "python/src", "python/run_*.py"],
            "Black 自動修正",
        )

        # isort 自動修正
        self.run_command(
            [
                self.python_cmd,
                "-m",
                "isort",
                "--profile=black",
                "--line-length=100",
                "python/src",
                "python/run_*.py",
            ],
            "isort 自動修正",
        )

        print("\n💡 ヒント: 修正後は再度チェックを実行してください")
        print("    python run_code_review.py")


def main():
    """メインエントリポイント"""
    import argparse

    parser = argparse.ArgumentParser(description="StockFixer コードレビュー実行スクリプト")
    parser.add_argument(
        "files",
        nargs="*",
        help="対象ファイル（省略時は全ファイル）",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="統合テスト + カバレッジを含む完全レビュー",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="自動修正を実行",
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="コード品質チェックのみ",
    )
    parser.add_argument(
        "--locks",
        action="store_true",
        help="ロック問題検出のみ",
    )
    parser.add_argument(
        "--tests",
        action="store_true",
        help="テストのみ実行",
    )

    args = parser.parse_args()

    # 実行対象ファイルの決定
    target_files = args.files if args.files else None

    # レビュー実行
    reviewer = CodeReviewRunner(verbose=True)

    if args.fix:
        # 自動修正モード
        reviewer.run_auto_fix()
    elif args.quality:
        # コード品質チェックのみ
        reviewer.run_code_quality_checks(target_files)
    elif args.locks:
        # ロック問題検出のみ
        reviewer.run_lock_detection(target_files)
    elif args.tests:
        # テストのみ
        reviewer.run_tests(full=args.full)
    else:
        # 通常レビュー（質+ロック+テスト）
        reviewer.run_code_quality_checks(target_files)
        reviewer.run_lock_detection(target_files)
        reviewer.run_tests(full=args.full)

    # 結果サマリー表示
    success = reviewer.print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
