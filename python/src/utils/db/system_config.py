"""system_config テーブルの操作 (R-410: ドリフト監視閾値の動的設定)"""

from typing import Optional

from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_config_value(key: str, default: Optional[str] = None) -> Optional[str]:
    """system_config テーブルから設定値を取得する。"""
    with _db_connection() as con:
        row = con.execute(
            "SELECT value FROM system_config WHERE key = ?", [key]
        ).fetchone()
    return row[0] if row else default


def set_config_value(key: str, value: str) -> None:
    """system_config テーブルに設定値を保存（UPSERT）する。"""
    with _db_connection() as con:
        con.execute(
            """
            INSERT INTO system_config (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
            """,
            [key, value],
        )
    logger.info("system_config 更新: %s = %s", key, value)
