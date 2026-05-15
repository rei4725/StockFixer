"""
DuckDB 定期バックアップパイプライン (NF-602)

DB ファイルをタイムスタンプ付きディレクトリにコピーし、最大5世代を保持する。
"""

import os
import shutil
import time
from datetime import datetime

from src.utils.data_path_utils import get_data_dir, get_db_path
from src.utils.db import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_GENERATIONS = 5


def get_backup_dir() -> str:
    """バックアップルートディレクトリのパスを返す"""
    return os.path.join(get_data_dir(), "backups")


def run_db_backup() -> dict:
    """
    DuckDB バックアップを実行する。

    手順:
        1. CHECKPOINT で WAL をメインファイルへフラッシュ
        2. タイムスタンプ付きディレクトリへファイルコピー
        3. 5世代超過分を古い順に削除

    Returns:
        dict: backup_path, size_mb, elapsed_seconds, pruned_count, error
    """
    db_path = get_db_path()
    backup_root = get_backup_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dest_dir = os.path.join(backup_root, timestamp)
    backup_dest_path = os.path.join(backup_dest_dir, "stockfixer.duckdb")

    start = time.monotonic()
    error_msg = None
    size_mb = 0.0
    pruned_count = 0

    try:
        # 1. CHECKPOINT（WAL をメインファイルへフラッシュしてコピー対象を確定させる）
        logger.info("バックアップ: CHECKPOINT 開始")
        with _db_connection() as con:
            con.execute("CHECKPOINT")
        logger.info("バックアップ: CHECKPOINT 完了")

        # 2. ファイルコピー
        os.makedirs(backup_dest_dir, exist_ok=True)
        logger.info("バックアップ: コピー開始 → %s", backup_dest_path)
        shutil.copy2(db_path, backup_dest_path)
        size_mb = os.path.getsize(backup_dest_path) / (1024 * 1024)
        logger.info("バックアップ: コピー完了 (%.2f MB)", size_mb)

        # 3. 古い世代を削除
        pruned_count = _prune_old_backups(backup_root, MAX_GENERATIONS)

    except Exception as e:
        logger.error("バックアップ失敗: %s", e, exc_info=True)
        error_msg = str(e)

    elapsed = time.monotonic() - start
    logger.info(
        "=== バックアップ完了: %.1f 秒, %.2f MB, 削除世代=%s ===",
        elapsed,
        size_mb,
        pruned_count,
    )

    return {
        "backup_path": backup_dest_path,
        "size_mb": size_mb,
        "elapsed_seconds": elapsed,
        "pruned_count": pruned_count,
        "error": error_msg,
    }


def _prune_old_backups(backup_root: str, max_generations: int) -> int:
    """max_generations を超えた古いバックアップを削除し、削除件数を返す。"""
    if not os.path.isdir(backup_root):
        return 0

    # YYYYMMDD_HHMMSS 形式のディレクトリのみ対象（辞書順 = 時系列順）
    entries = sorted(
        e for e in os.listdir(backup_root) if os.path.isdir(os.path.join(backup_root, e))
    )
    to_delete = entries[: max(0, len(entries) - max_generations)]

    for name in to_delete:
        target = os.path.join(backup_root, name)
        try:
            shutil.rmtree(target)
            logger.info("古いバックアップを削除: %s", target)
        except Exception as e:
            logger.error("バックアップ削除失敗 (%s): %s", target, e, exc_info=True)

    return len(to_delete)
