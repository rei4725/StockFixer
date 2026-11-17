"""data_quality_log テーブル操作モジュール"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

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
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                [market, symbol, item["check"], item["level"], item["detail"], checked_at],
            )
    logger.debug(f"品質ログ記録: {market}/{symbol} {len(issues)}件")


@dataclass(frozen=True)
class QualityLogRecord:
    """data_quality_log の1行（読み返す分。#1951）。checked_at は書いたときの ISO 8601 の文字列。"""

    market: str
    symbol: str
    check_name: str
    level: str
    detail: str
    checked_at: str


def _to_utc_iso(value: datetime) -> str:
    """checked_at と比べるため、日時を UTC の ISO 8601 の文字列にする（tz の無い値は UTC とみなす）。"""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def load_quality_logs(
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    market: Optional[str] = None,
) -> List[QualityLogRecord]:
    """data_quality_log を checked_at の順に返す（読むだけ。#1951）。

    since を渡すと checked_at がそれ以後、until を渡すとそれより前の行だけを返す。
    market を渡すとその市場だけを返す。
    """
    conditions: list[str] = []
    params: list[object] = []
    if since is not None:
        conditions.append("checked_at >= %s")
        params.append(_to_utc_iso(since))
    if until is not None:
        conditions.append("checked_at < %s")
        params.append(_to_utc_iso(until))
    if market is not None:
        conditions.append("market = %s")
        params.append(market)
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = (
        "SELECT market, symbol, check_name, level, detail, checked_at "
        "FROM data_quality_log" + where + " ORDER BY checked_at, market, symbol, check_name"
    )
    with _db_connection() as con:
        rows = con.execute(sql, params).fetchall()
    return [
        QualityLogRecord(
            market=r[0],
            symbol=r[1],
            check_name=r[2],
            level=r[3],
            detail=r[4],
            checked_at=r[5],
        )
        for r in rows
    ]
