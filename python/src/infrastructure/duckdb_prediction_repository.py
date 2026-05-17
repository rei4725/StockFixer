"""DuckDB 予測結果リポジトリ

PredictionResultRepository ポートの DuckDB 実装。
既存の src/utils/db/prediction.py 関数に委譲する。
"""

import pandas as pd

from src.domain.ports import PredictionResultRepository
from src.utils.db.prediction import load_prediction_results, save_prediction_results
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DuckDBPredictionRepository(PredictionResultRepository):
    """DuckDB を使った予測結果の永続化アダプター"""

    def save(self, predicted_at: str, results: list) -> None:
        save_prediction_results(predicted_at, results)

    def load(self, market: str, limit: int = 100) -> pd.DataFrame:
        return load_prediction_results(market=market, limit=limit)
