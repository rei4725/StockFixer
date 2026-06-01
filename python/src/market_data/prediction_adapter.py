"""
PredictionMarketDataAdapter - MarketDataPort の具象実装。

market_data BC の実際のモジュールを MarketDataPort インターフェースに適合させる。
market_data BC 内に配置することで prediction -> market_data 直接依存を最小化する。
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import pandas as pd


class PredictionMarketDataAdapter:
    """market_data BC の関数を MarketDataPort インターフェースに適合させるアダプター。"""

    def get_earnings_dates(self, market: str, symbol: str) -> pd.DatetimeIndex:
        from src.market_data.loader import get_earnings_dates

        return get_earnings_dates(market, symbol)

    def add_earnings_flag(
        self,
        df: pd.DataFrame,
        earnings_dates: Union[pd.DatetimeIndex, list, tuple],
        lookaround_days: int = 3,
    ) -> pd.DataFrame:
        from src.market_data.technical import add_earnings_flag

        return add_earnings_flag(df, earnings_dates, lookaround_days=lookaround_days)

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        from src.market_data.technical import add_technical_indicators

        return add_technical_indicators(df)

    def create_basic_lag_features(
        self,
        df: pd.DataFrame,
        n_lags: int = 10,
        feature_cols: Optional[List[str]] = None,
        target_horizon: int = 1,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        from src.market_data.technical import create_basic_lag_features

        return create_basic_lag_features(
            df, n_lags=n_lags, feature_cols=feature_cols, target_horizon=target_horizon
        )
