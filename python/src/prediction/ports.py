"""
MarketDataPort - market_data BC への依存を逆転させるポート定義。

prediction BC はこのポートを通じて market_data BC の機能を利用する。
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Tuple, Union, runtime_checkable

import pandas as pd


@runtime_checkable
class MarketDataPort(Protocol):
    """prediction 用 market_data アクセスポートのインターフェース。"""

    def get_earnings_dates(self, market: str, symbol: str) -> pd.DatetimeIndex: ...

    def add_earnings_flag(
        self,
        df: pd.DataFrame,
        earnings_dates: Union[pd.DatetimeIndex, list, tuple],
        lookaround_days: int = 3,
    ) -> pd.DataFrame: ...

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame: ...

    def create_basic_lag_features(
        self,
        df: pd.DataFrame,
        n_lags: int = 10,
        feature_cols: Optional[List[str]] = None,
        target_horizon: int = 1,
    ) -> Tuple[pd.DataFrame, pd.Series]: ...


_port: Optional[MarketDataPort] = None


def set_market_data_port(port: MarketDataPort) -> None:
    """テストや orchestration から実装を注入するためのセッター。"""
    global _port
    _port = port


def get_market_data_port() -> MarketDataPort:
    """デフォルト実装（PredictionMarketDataAdapter）を遅延初期化して返す。"""
    global _port
    if _port is None:
        from src.market_data.prediction_adapter import PredictionMarketDataAdapter

        _port = PredictionMarketDataAdapter()
    return _port
