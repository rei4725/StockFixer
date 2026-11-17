"""朝の目線の告知の記録を溜める表 morning_view_log の保存と読み出し（#1875 段2）。

DDL はマイグレーション 0012 が作る。1日1行で、同じ日は上書きする。
読み出しは段1の ViewRecord の並びで返し、morning_view.tally にそのまま渡せる。
"""

from datetime import date, datetime, timezone

from src.reporting.morning_view import ViewRecord
from src.utils.db import db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def save_view_record(record: ViewRecord) -> None:
    """告知した1日分を保存する。同じ日の行があれば上書きする。"""
    with db_connection() as con:
        con.execute(
            """
            INSERT INTO morning_view_log
                (day, view, source, futures_view, actual_change_pct, announced_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (day) DO UPDATE SET
                view = EXCLUDED.view,
                source = EXCLUDED.source,
                futures_view = EXCLUDED.futures_view,
                actual_change_pct = EXCLUDED.actual_change_pct,
                announced_at = EXCLUDED.announced_at
            """,
            [
                record.day,
                record.view,
                record.source,
                record.futures_view,
                record.actual_change_pct,
                _utc_now(),
            ],
        )
    logger.info("morning_view_log 保存: %s %s (%s)", record.day, record.view, record.source)


def record_view_outcome(day: date, actual_change_pct: float) -> bool:
    """答え合わせの騰落率を書き足す。その日の告知が無ければ何もせず False を返す。"""
    with db_connection() as con:
        row = con.execute("SELECT 1 FROM morning_view_log WHERE day = %s", [day]).fetchone()
        if row is None:
            logger.info("morning_view_log に %s の告知が無いので答え合わせしない", day)
            return False
        con.execute(
            "UPDATE morning_view_log SET actual_change_pct = %s, judged_at = %s WHERE day = %s",
            [actual_change_pct, _utc_now(), day],
        )
    logger.info("morning_view_log 答え合わせ: %s %+.2f%%", day, actual_change_pct)
    return True


def load_view_records() -> list[ViewRecord]:
    """全行を日の順に ViewRecord の並びで返す。"""
    with db_connection() as con:
        rows = con.execute(
            "SELECT day, view, source, futures_view, actual_change_pct "
            "FROM morning_view_log ORDER BY day"
        ).fetchall()
    return [
        ViewRecord(
            day=row[0] if isinstance(row[0], date) else date.fromisoformat(str(row[0])),
            view=str(row[1]),
            source=str(row[2]),
            futures_view=str(row[3]),
            actual_change_pct=None if row[4] is None else float(row[4]),
        )
        for row in rows
    ]
