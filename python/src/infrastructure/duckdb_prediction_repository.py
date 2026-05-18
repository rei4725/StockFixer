"""DuckDB 予測結果リポジトリ

PredictionResultRepository ポートの DuckDB 実装。
既存の src/utils/db/prediction.py 関数に委譲する。
"""

import pandas as pd

from src.domain.ports import PredictionResultRepository
from src.utils.db._connection import _db_connection
from src.utils.db.prediction import load_prediction_results, save_prediction_results
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DuckDBPredictionRepository(PredictionResultRepository):
    """DuckDB を使った予測結果の永続化アダプター"""

    def save(self, predicted_at: str, results: list) -> None:
        save_prediction_results(predicted_at, results)

    def load(self, market: str, limit: int = 100) -> pd.DataFrame:
        return load_prediction_results(market=market, limit=limit)

    def get_latest_by_market(self, market: str) -> pd.DataFrame:
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
