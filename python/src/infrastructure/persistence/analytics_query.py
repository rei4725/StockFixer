"""reporting 向けの読み取り専用クエリのアダプタ。"""

from datetime import datetime, timedelta

from src.domain.ports import AnalyticsQuery
from src.utils.db import db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PostgresAnalyticsQuery(AnalyticsQuery):
    """Postgres から reporting 向けの集計を読む AnalyticsQuery 実装。"""

    def paper_real_diff_summary(self, recent_days: int = 7) -> dict:
        since = datetime.now() - timedelta(days=recent_days)
        with db_connection() as con:
            row = con.execute(
                """
            SELECT
                COUNT(*) AS tracked_count,
                COUNT(*) FILTER (
                    WHERE paper_price IS NOT NULL AND real_price IS NOT NULL
                ) AS comparable_count,
                AVG(paper_slippage) AS avg_paper_slippage,
                AVG(real_slippage) AS avg_real_slippage,
                AVG(ABS(price_diff)) AS avg_abs_price_diff,
                AVG(ABS(price_diff / NULLIF(signal_price, 0))) AS avg_abs_diff_ratio,
                MAX(ABS(price_diff)) AS max_abs_price_diff
            FROM paper_real_diff
            WHERE COALESCE(paper_filled_at, real_checked_at, created_at) >= %s
            """,
                [since],
            ).fetchone()

        return {
            "tracked_count": int(row[0] or 0),
            "comparable_count": int(row[1] or 0),
            "avg_paper_slippage": float(row[2] or 0.0),
            "avg_real_slippage": float(row[3] or 0.0),
            "avg_abs_price_diff": float(row[4] or 0.0),
            "avg_abs_diff_ratio": float(row[5] or 0.0),
            "max_abs_price_diff": float(row[6] or 0.0),
        }
