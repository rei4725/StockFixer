#!/usr/bin/env python
"""
テストファイル整理スクリプト
Unit Test / Integration Test に振り分けて移動する
"""
import shutil
import os

# Unit Test に移動するファイル（依存度が低い）
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

# Integration Test に移動するファイル（DB・外部依存あり）
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

os.chdir("tests")

print("=" * 60)
print("テストファイル整理開始")
print("=" * 60)

# Unit Test を移動
print("\n[Unit Test に移動]")
for fname in unit_tests:
    src = fname
    dst = os.path.join("unit", fname)
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"  ✓ {fname}")
    else:
        print(f"  - {fname} (見つかりません)")

# Integration Test を移動
print("\n[Integration Test に移動]")
for fname in integration_tests:
    src = fname
    dst = os.path.join("integration", fname)
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"  ✓ {fname}")
    else:
        print(f"  - {fname} (見つかりません)")

print("\n" + "=" * 60)
print("完了")
print("=" * 60)
