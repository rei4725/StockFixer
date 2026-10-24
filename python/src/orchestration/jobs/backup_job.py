"""日次バックアップジョブ。

daily.py からの責務分離（ファイル行数ガード対策）。daily.py から re-export する。
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_daily_backup() -> None:
    """
    毎日深夜実行: PostgreSQL を pg_dump（カスタムフォーマット）でタイムスタンプ付き
    ディレクトリへ出力し、最大5世代を保持する。

    手順:
        1. pg_dump（-Fc）で data/backups/YYYYMMDD_HHMMSS/stockfixer.dump へ出力
        2. 5世代超過分を古い順に削除
        3. Discord に完了通知
    """
    from src.orchestration.backup_pipeline import run_db_backup
    from src.reporting.discord.discord_utils import send_backup_completion

    logger.info("=== 日次バックアップ開始 ===")
    result = run_db_backup()

    try:
        send_backup_completion(
            backup_path=result["backup_path"],
            size_mb=result["size_mb"],
            elapsed_seconds=result["elapsed_seconds"],
            pruned_count=result["pruned_count"],
            error=result["error"],
        )
    except Exception as e:
        logger.error("バックアップ通知失敗: %s", e, exc_info=True)

    if result["error"]:
        raise RuntimeError(f"バックアップ失敗: {result['error']}")

    logger.info("=== 日次バックアップ完了 ===")
