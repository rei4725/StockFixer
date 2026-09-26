"""prediction_results 読み取りポートの Postgres アダプタ。

prediction_results は prediction BC 自身も読み書きする所有テーブルであるため、SQL は
src.prediction.db.prediction_results に残し、ここでは委譲するだけに留める（SQL を二重化しない）。
"""

from typing import Optional

import pandas as pd

from src.domain.ports import PredictionResultRepository
from src.prediction.db import prediction_results as _results


class PostgresPredictionResultRepository(PredictionResultRepository):
    """Postgres の prediction_results を読む PredictionResultRepository 実装。"""

    def latest_timestamp(self, model_version: Optional[str] = None) -> Optional[str]:
        return _results.load_latest_prediction_timestamp(model_version=model_version)

    def results_at(
        self,
        predicted_at: Optional[str] = None,
        market: Optional[str] = None,
        top_n: Optional[int] = None,
        worst_n: Optional[int] = None,
    ) -> pd.DataFrame:
        return _results.load_prediction_results(
            predicted_at=predicted_at, market=market, top_n=top_n, worst_n=worst_n
        )

    def markets_at(self, predicted_at: Optional[str] = None) -> list:
        return _results.load_prediction_markets(predicted_at=predicted_at)

    def get_latest_by_market(self, market: str) -> pd.DataFrame:
        return _results.load_latest_predictions_by_market(market)
