"""order_run_summary テーブル: 発注実行サマリー（R-214）のアダプタ。"""

from src.domain.ports import OrderRunSink
from src.domain.types import OrderRunSummary
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PostgresOrderRunSink(OrderRunSink):
    """order_run_summary テーブルへ書き込む OrderRunSink 実装。"""

    def save(self, summary: OrderRunSummary) -> None:
        with _db_connection() as con:
            con.execute(
                """
                INSERT INTO order_run_summary
                    (run_id, market, mode, run_at, buy_orders, sell_orders, short_orders,
                     skipped, skipped_min_change, total_turnover, min_change_ratio)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    summary.run_id,
                    summary.market,
                    summary.mode,
                    summary.buy_orders,
                    summary.sell_orders,
                    summary.short_orders,
                    summary.skipped,
                    summary.skipped_min_change,
                    summary.total_turnover,
                    summary.min_change_ratio,
                ],
            )
        logger.info(
            f"order_run_summary 保存: run_id={summary.run_id} market={summary.market} "
            f"mode={summary.mode} "
            f"buy={summary.buy_orders} sell={summary.sell_orders} short={summary.short_orders} "
            f"skipped={summary.skipped}(min_change={summary.skipped_min_change}) "
            f"turnover={summary.total_turnover:.0f}"
        )
