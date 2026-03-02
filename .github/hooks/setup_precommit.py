#!/usr/bin/env python3
"""
Pre-commit hooks セットアップスクリプト

初回セットアップ時に実行。見立てのチェック・インストール・設定を自動化する。
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, shell=True, check=True):
    """コマンドを実行し、失敗時は例外を発生させる"""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True, check=check)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)


def check_python_version():
    """Python 3.8+ をチェック"""
    print("[1/5] Python バージョン確認 ...", end=" ")
    if sys.version_info >= (3, 8):
        version_str = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(f"✓ (Python {version_str})")
        return True
    else:
        print(f"✗ (Python {sys.version_info.major}.{sys.version_info.minor} - 3.8以上必須)")
        return False


def check_pip():
    """pip が使用可能か確認"""
    print("[2/5] pip 確認 ...", end=" ")
    success, stdout, stderr = run_command(f"{sys.executable} -m pip --version", check=False)
    if success:
        version = stdout.split()[1] if stdout else "unknown"
        print(f"✓ ({version})")
        return True
    else:
        print("✗")
        return False


def install_dependencies():
    """依存パッケージをインストール"""
    print("[3/5] 依存パッケージをインストール ...", end=" ")
    
    # requirements.txt の場所を特定
    script_dir = Path(__file__).parent.parent.parent  # .github/hooks/ → StockFixer
    req_file = script_dir / "python" / "requirements.txt"
    
    if not req_file.exists():
        print(f"✗ (requirements.txt が見つかりません: {req_file})")
        return False
    
    success, stdout, stderr = run_command(
        f"{sys.executable} -m pip install -q -r {req_file}",
        check=False
    )
    
    if success:
        print("✓")
        return True
    else:
        print(f"✗\n詳細: {stderr}")
        return False


def setup_git_hooks():
    """Git pre-commit hooksをセットアップ"""
    print("[4/5] Git pre-commit hooksをセットアップ ...", end=" ")
    
    # プロジェクトルートを特定
    script_dir = Path(__file__).parent.parent.parent  # .github/hooks/ → StockFixer
    
    success1, _, stderr1 = run_command("pre-commit install", check=False)
    success2, _, stderr2 = run_command("pre-commit install --hook-type commit-msg", check=False)
    
    if success1 and success2:
        print("✓")
        return True
    else:
        errors = [e for e in [stderr1, stderr2] if e]
        print(f"⚠️ (部分的に完了)\n詳細: {' / '.join(errors)}")
        return True  # 部分成功では続行


def verify_setup():
    """セットアップからが完了したか確認"""
    print("[5/5] セットアップ確認 ...", end=" ")
    
    success, stdout, stderr = run_command("pre-commit --version", check=False)
    if success:
        print(f"✓ ({stdout})")
        return True
    else:
        print("✗")
        return False


def main():
    print("\n" + "="*60)
    print("StockFixer Pre-commit Hooks セットアップ")
    print("="*60 + "\n")
    
    checks = [
        ("Python バージョン", check_python_version),
        ("pip", check_pip),
        ("依存パッケージ", install_dependencies),
        ("Git hooks", setup_git_hooks),
        ("セットアップ確認", verify_setup),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"エラー: {e}")
            results.append((name, False))
    
    # サマリー
    print("\n" + "="*60)
    print("セットアップサマリー")
    print("="*60)
    
    success_count = sum(1 for _, result in results if result)
    print(f"成功: {success_count}/{len(results)}")
    
    if success_count == len(results):
        print("\n✓ セットアップ完了！\n")
        print("次のコマンドで確認できます:")
        print("  pre-commit run --all-files")
        return 0
    else:
        print("\n✗ セットアップ未完了。詳細をご確認ください。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
