#!/usr/bin/env python
import os
import shutil
import sys

os.chdir(r'c:\src\StockFixer\python\tests')

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

log = []
log.append(f"Current dir: {os.getcwd()}")
log.append(f"Files before: {os.listdir('.')}")

# Move unit tests
for fname in unit_tests:
    src = os.path.join('.', fname)
    dst = os.path.join('unit', fname)
    try:
        if os.path.exists(src):
            shutil.move(src, dst)
            log.append(f"✓ Moved unit: {fname}")
        else:
            log.append(f"✗ Unit not found: {fname}")
    except Exception as e:
        log.append(f"✗ Error moving {fname}: {e}")

# Move integration tests
for fname in integration_tests:
    src = os.path.join('.', fname)
    dst = os.path.join('integration', fname)
    try:
        if os.path.exists(src):
            shutil.move(src, dst)
            log.append(f"✓ Moved integration: {fname}")
        else:
            log.append(f"✗ Integration not found: {fname}")
    except Exception as e:
        log.append(f"✗ Error moving {fname}: {e}")

log.append(f"Files after: {os.listdir('.')}")
log.append(f"Unit files: {os.listdir('unit')}")
log.append(f"Integration files: {os.listdir('integration')}")

output = "\n".join(log)
print(output)

# Write to file
with open('organize_log.txt', 'w') as f:
    f.write(output)
