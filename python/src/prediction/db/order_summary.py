"""order_run_summary テーブル: 発注実行サマリー（R-214）。"""

import pandas as pd

from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_order_run_summary(
    run_id: str,
    market: str,
    mode: str,
    buy_orders: int,
    sell_orders: int,
    short_orders: int,
    skipped: int,
    skipped_min_change: int,
    total_turnover: float,
    min_change_ratio: float,
) -> None:
    """発注実行サマリーを order_run_summary テーブルに保存する。"""
    with _db_connection() as con:
        con.execute(
            """
            INSERT INTO order_run_summary
                (run_id, market, mode, run_at, buy_orders, sell_orders, short_orders,
                 skipped, skipped_min_change, total_turnover, min_change_ratio)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                run_id,
                market,
                mode,
                buy_orders,
                sell_orders,
                short_orders,
                skipped,
                skipped_min_change,
                total_turnover,
                min_change_ratio,
            ],
        )
    logger.info(
        f"order_run_summary 保存: run_id={run_id} market={market} mode={mode} "
        f"buy={buy_orders} sell={sell_orders} short={short_orders} "
        f"skipped={skipped}(min_change={skipped_min_change}) turnover={total_turnover:.0f}"
    )


def load_turnover_comparison(market: str, limit: int = 30) -> pd.DataFrame:
    """
    order_run_summary から直近の売買代金推移を取得する。

    Args:
        market: マーケット識別子
        limit: 取得件数上限

    Returns:
        pd.DataFrame: [run_id, market, mode, run_at, total_turnover, buy_orders, sell_orders,
                        short_orders, skipped_min_change]
    """
    with _db_connection() as con:
        try:
            return pd.read_sql(
                """
                SELECT run_id, market, mode, run_at, total_turnover,
                       buy_orders, sell_orders, short_orders, skipped_min_change
                FROM order_run_summary
                WHERE market = %s
                ORDER BY run_at DESC
                LIMIT %s
                """,
                con,
                params=[market, limit],
            )
        except Exception as e:
            logger.error(f"load_turnover_comparison 失敗: {e}", exc_info=True)
            return pd.DataFrame()
