#!/usr/bin/env python3
"""テストファイルを unit/ と integration/ に移動する統括スクリプト"""
import os
import shutil
from pathlib import Path

# ユニットテスト（外部依存なし、Mock のみ）
unit_tests = [
    "test_backtester.py",
    "test_backtest_advanced.py",
    "test_backtest_metrics.py",
    "test_backtest_task.py",
    "test_data_path_utils.py",
    "test_df_to_string.py",
    "test_model_manager.py",
    "test_signal_generator.py",
    "test_technical_analysis.py",
]

# 統合テスト（DB、yfinance、外部API 依存）
integration_tests = [
    "test_batch_runner.py",
    "test_data_loader.py",
    "test_data_pipeline.py",
    "test_db.py",
    "test_db_raw.py",
    "test_discord_bot.py",
    "test_model_training_pipeline.py",
    "test_prediction_pipeline.py",
    "test_scheduler_pipeline.py",
]

tests_dir = Path(__file__).parent / "python" / "tests"
unit_dir = tests_dir / "unit"
integration_dir = tests_dir / "integration"

# ディレクトリ確認
assert tests_dir.exists(), f"テストディレクトリが見つかりません: {tests_dir}"
assert unit_dir.exists(), f"unit ディレクトリが見つかりません: {unit_dir}"
assert integration_dir.exists(), f"integration ディレクトリが見つかりません: {integration_dir}"

# ユニットテストを移動
print(f"=== ユニットテスト ({len(unit_tests)} ファイル) を {unit_dir.relative_to(tests_dir)} に移動 ===")
for fname in unit_tests:
    src = tests_dir / fname
    dst = unit_dir / fname
    if src.exists():
        try:
            shutil.move(str(src), str(dst))
            print(f"✓ {fname}")
        except Exception as e:
            print(f"✗ {fname}: {e}")
    else:
        print(f"- {fname} (見つかりません)")

# 統合テストを移動
print(f"\n=== 統合テスト ({len(integration_tests)} ファイル) を {integration_dir.relative_to(tests_dir)} に移動 ===")
for fname in integration_tests:
    src = tests_dir / fname
    dst = integration_dir / fname
    if src.exists():
        try:
            shutil.move(str(src), str(dst))
            print(f"✓ {fname}")
        except Exception as e:
            print(f"✗ {fname}: {e}")
    else:
        print(f"- {fname} (見つかりません)")

# 結果確認
print("\n=== 移動結果 ===")
unit_files = sorted([f.name for f in unit_dir.glob("*.py") if f.is_file()])
integration_files = sorted([f.name for f in integration_dir.glob("*.py") if f.is_file()])
root_files = sorted([f.name for f in tests_dir.glob("test_*.py") if f.is_file()])

print(f"unit/ に移動: {len(unit_files)} ファイル")
for f in unit_files:
    print(f"  - {f}")

print(f"\nintegration/ に移動: {len(integration_files)} ファイル")
for f in integration_files:
    print(f"  - {f}")

if root_files:
    print(f"\n⚠️ tests/ ルートに残っているテストファイル: {len(root_files)} ファイル")
    for f in root_files:
        print(f"  - {f}")
else:
    print(f"\n✓ tests/ ルートにはテストファイルが残っていません")

print("\n完了")
