"""order_run_summary テーブル: 発注実行サマリー（R-214）のアダプタ。"""

from datetime import datetime
from typing import List, Optional

from src.domain.ports import OrderRunSink
from src.domain.types import OrderRunSummary
from src.reporting.order_run_trend import OrderRunRecord
from src.utils.db import db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PostgresOrderRunSink(OrderRunSink):
    """order_run_summary テーブルへ書き込む OrderRunSink 実装。"""

    def save(self, summary: OrderRunSummary) -> None:
        with db_connection() as con:
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


def load_order_run_summaries(
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    market: Optional[str] = None,
) -> List[OrderRunRecord]:
    """order_run_summary を run_at の順に返す（読むだけ。#1981）。

    since を渡すと run_at がそれ以後、until を渡すとそれより前の行だけを返す。
    market を渡すとその市場だけを返す。mode では絞らない。
    """
    conditions: list[str] = []
    params: list[object] = []
    if since is not None:
        conditions.append("run_at >= %s")
        params.append(since)
    if until is not None:
        conditions.append("run_at < %s")
        params.append(until)
    if market is not None:
        conditions.append("market = %s")
        params.append(market)
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = (
        "SELECT run_id, market, mode, run_at, buy_orders, sell_orders, short_orders, "
        "skipped, skipped_min_change, total_turnover, min_change_ratio "
        "FROM order_run_summary" + where + " ORDER BY run_at, market, run_id"
    )
    with db_connection() as con:
        rows = con.execute(sql, params).fetchall()
    return [
        OrderRunRecord(
            run_id=r[0],
            market=r[1],
            mode=r[2],
            run_at=r[3],
            buy_orders=int(r[4]),
            sell_orders=int(r[5]),
            short_orders=int(r[6]),
            skipped=int(r[7]),
            skipped_min_change=int(r[8]),
            total_turnover=float(r[9]),
            min_change_ratio=None if r[10] is None else float(r[10]),
        )
        for r in rows
    ]
