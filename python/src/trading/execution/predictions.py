"""予測結果のロード。"""

import pandas as pd

from src.utils.db._connection import _db_connection


def _load_latest_predictions(market: str) -> pd.DataFrame:
    """
    DuckDB から当日の最新予測結果を取得する。

    Returns:
        columns: market, symbol, predicted_at, current_price, diff_ratio,
        confidence_ratio, diff_ratio_3d, diff_ratio_5d, diff_ratio_10d,
        confluence_score (desc order by diff_ratio)
    """
    with _db_connection() as con:
        return con.execute(
            """
            WITH latest AS (
                SELECT market, symbol, MAX(predicted_at) AS latest_at
                FROM prediction_results
                WHERE market = ?
                GROUP BY market, symbol
            )
            SELECT
                pr.market,
                pr.symbol,
                pr.predicted_at,
                pr.current_price,
                pr.diff_ratio,
                pr.confidence_ratio,
                pr.diff_ratio_3d,
                pr.diff_ratio_5d,
                pr.diff_ratio_10d,
                pr.confluence_score
            FROM prediction_results pr
            JOIN latest l
              ON pr.market = l.market AND pr.symbol = l.symbol AND pr.predicted_at = l.latest_at
            WHERE pr.diff_ratio IS NOT NULL
            ORDER BY pr.diff_ratio DESC
            """,
            [market],
        ).df()
