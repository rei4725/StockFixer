"""
DuckDB バックアップ CLI エントリポイント

使い方:
  py run_backup.py
"""

import sys

from src.orchestration.backup_pipeline import run_db_backup
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    logger.info("=== 手動バックアップ開始 ===")
    result = run_db_backup()
    if result["error"]:
        logger.error("バックアップ失敗: %s", result["error"])
        sys.exit(1)
    print(
        f"バックアップ完了: {result['backup_path']} "
        f"({result['size_mb']:.2f} MB, {result['elapsed_seconds']:.1f} 秒)"
    )


if __name__ == "__main__":
    main()
