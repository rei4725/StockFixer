"""
PostgreSQL 定期バックアップパイプライン (NF-602)

pg_dump（カスタムフォーマット）でタイムスタンプ付きファイルへ出力し、最大5世代を保持する。
"""

import os
import shutil
import subprocess
import time
from datetime import datetime
from urllib.parse import urlparse

from src.utils.data_path_utils import get_data_dir, get_database_url
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_GENERATIONS = 5
_PG_DUMP_TIMEOUT_SECONDS = 300


def get_backup_dir() -> str:
    """バックアップルートディレクトリのパスを返す"""
    return os.path.join(get_data_dir(), "backups")


def run_db_backup() -> dict:
    """
    PostgreSQL バックアップを実行する。

    手順:
        1. pg_dump（カスタムフォーマット, -Fc）でタイムスタンプ付きファイルへ出力
        2. 5世代超過分を古い順に削除

    Returns:
        dict: backup_path, size_mb, elapsed_seconds, pruned_count, error
    """
    backup_root = get_backup_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dest_dir = os.path.join(backup_root, timestamp)
    backup_dest_path = os.path.join(backup_dest_dir, "stockfixer.dump")

    start = time.monotonic()
    error_msg = None
    size_mb = 0.0
    pruned_count = 0
    # 失敗時に消すのはこの回に作ったフォルダだけ（同じ秒の既存フォルダは消さない）
    created_dest_dir = not os.path.isdir(backup_dest_dir)

    try:
        os.makedirs(backup_dest_dir, exist_ok=True)
        logger.info("バックアップ: pg_dump 開始 → %s", backup_dest_path)
        _run_pg_dump(get_database_url(), backup_dest_path)
        size_mb = os.path.getsize(backup_dest_path) / (1024 * 1024)
        logger.info("バックアップ: pg_dump 完了 (%.2f MB)", size_mb)

        pruned_count = _prune_old_backups(backup_root, MAX_GENERATIONS)

    except Exception as e:
        logger.error("バックアップ失敗: %s", e, exc_info=True)
        error_msg = str(e)
        if created_dest_dir:
            _remove_failed_backup_dir(backup_dest_dir)

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


def _run_pg_dump(database_url: str, dest_path: str) -> None:
    """pg_dump をカスタムフォーマットで実行する（環境変数でパスワードを渡し、コマンドラインへの露出を避ける）。"""
    parsed = urlparse(database_url)
    env = dict(os.environ)
    if parsed.password:
        env["PGPASSWORD"] = parsed.password

    cmd = [
        "pg_dump",
        "-Fc",
        "-h",
        parsed.hostname or "localhost",
        "-p",
        str(parsed.port or 5432),
        "-U",
        parsed.username or "",
        "-f",
        dest_path,
        (parsed.path or "/").lstrip("/"),
    ]
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=_PG_DUMP_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pg_dump 失敗 (code={result.returncode}): {result.stderr}")


def _remove_failed_backup_dir(backup_dest_dir: str) -> None:
    """失敗した回のフォルダ（書きかけの dump を含む）を消す。"""
    if not os.path.isdir(backup_dest_dir):
        return
    try:
        shutil.rmtree(backup_dest_dir)
        logger.info("失敗したバックアップのフォルダを削除: %s", backup_dest_dir)
    except Exception as e:
        logger.error(
            "失敗したバックアップのフォルダ削除失敗 (%s): %s", backup_dest_dir, e, exc_info=True
        )


def _has_valid_dump(backup_dir: str) -> bool:
    """stockfixer.dump があり大きさが0より大きいかを返す。"""
    dump_path = os.path.join(backup_dir, "stockfixer.dump")
    return os.path.isfile(dump_path) and os.path.getsize(dump_path) > 0


def _prune_old_backups(backup_root: str, max_generations: int) -> int:
    """
    max_generations を超えた古いバックアップと失敗の跡を削除し、削除件数を返す。

    世代に数えるのは stockfixer.dump があり大きさが0より大きいフォルダだけ。
    dump が無いか大きさ0のフォルダは失敗の跡として削除する。
    """
    if not os.path.isdir(backup_root):
        return 0

    # YYYYMMDD_HHMMSS 形式のディレクトリのみ対象（辞書順 = 時系列順）
    entries = sorted(
        e for e in os.listdir(backup_root) if os.path.isdir(os.path.join(backup_root, e))
    )
    valid = [e for e in entries if _has_valid_dump(os.path.join(backup_root, e))]
    leftovers = [e for e in entries if e not in valid]
    to_delete = leftovers + valid[: max(0, len(valid) - max_generations)]

    for name in to_delete:
        target = os.path.join(backup_root, name)
        try:
            shutil.rmtree(target)
            logger.info("古いバックアップを削除: %s", target)
        except Exception as e:
            logger.error("バックアップ削除失敗 (%s): %s", target, e, exc_info=True)

    return len(to_delete)
