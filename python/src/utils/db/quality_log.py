"""data_quality_log テーブル操作モジュール"""

from datetime import datetime, timezone
from typing import List

from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def insert_quality_log(
    market: str,
    symbol: str,
    issues: List[dict],
) -> None:
    """品質チェック結果を data_quality_log テーブルに挿入する。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
        issues: [{"check": str, "level": str, "detail": str}] のリスト
    """
    if not issues:
        return
    checked_at = datetime.now(timezone.utc).isoformat()
    with _db_connection() as con:
        for item in issues:
            con.execute(
                """
                INSERT INTO data_quality_log
                    (market, symbol, check_name, level, detail, checked_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [market, symbol, item["check"], item["level"], item["detail"], checked_at],
            )
    logger.debug(f"品質ログ記録: {market}/{symbol} {len(issues)}件")
