"""DuckDB 株式特徴量リポジトリ

StockFeatureRepository ポートの DuckDB 実装。
既存の src/utils/db/stock_features.py 関数に委譲する。
"""

from typing import Optional

import pandas as pd

from src.domain.ports import StockFeatureRepository
from src.utils.db.stock_features import (
    get_all_symbols,
    load_all_stock_features,
    load_stock_features,
    upsert_stock_features,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DuckDBStockFeatureRepository(StockFeatureRepository):
    """DuckDB を使った株式特徴量の永続化アダプター"""

    def upsert(self, market: str, symbol: str, df: pd.DataFrame) -> None:
        upsert_stock_features(market, symbol, df)

    def load(self, market: str, symbol: str) -> Optional[pd.DataFrame]:
        return load_stock_features(market, symbol)

    def load_all(self) -> pd.DataFrame:
        return load_all_stock_features()

    def get_all_symbols(self) -> list:
        return get_all_symbols()
