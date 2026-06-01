"""
BacktestMarketDataAdapter - BacktestDataPort の具象実装。

market_data BC の実際のモジュールを BacktestDataPort インターフェースに適合させる。
market_data BC 内に配置することで backtest -> market_data 直接依存を最小化する。
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import pandas as pd


class BacktestMarketDataAdapter:
    """market_data BC の関数を BacktestDataPort インターフェースに適合させるアダプター。"""

    def get_stock_data(
        self, market: str, ticker: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        from src.market_data.loader import get_stock_data

        return get_stock_data(market, ticker, start, end)

    def get_raw_ohlcv_from_db(
        self, market: str, symbol: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        from src.market_data.loader import get_raw_ohlcv_from_db

        return get_raw_ohlcv_from_db(market, symbol, start, end)

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        from src.market_data.technical import add_technical_indicators

        return add_technical_indicators(df)

    def create_basic_lag_features(
        self, df: pd.DataFrame, n_lags: int = 10
    ) -> Tuple[pd.DataFrame, pd.Series]:
        from src.market_data.technical import create_basic_lag_features

        return create_basic_lag_features(df, n_lags=n_lags)

    def download(
        self,
        ticker: str,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        from src.market_data import yf_client

        return yf_client.download(ticker, start=start, end=end)

    def classify_regime(self, df: pd.DataFrame, **kwargs: Any) -> pd.Series:
        from src.market_data.technical import classify_regime

        return classify_regime(df, **kwargs)

    def get_market_regime(self, proxy_df: pd.DataFrame) -> pd.Series:
        from src.market_data.market_regime import get_market_regime

        return get_market_regime(proxy_df)
