"""
BacktestDataPort - market_data BC への依存を逆転させるポート定義。

backtest BC はこのポートを通じて market_data BC の機能を利用する。
"""

from __future__ import annotations

from typing import Optional, Protocol, Tuple, runtime_checkable

import pandas as pd


@runtime_checkable
class BacktestDataPort(Protocol):
    """バックテスト用 market_data アクセスポートのインターフェース。"""

    def get_stock_data(
        self, market: str, ticker: str, start: str, end: str
    ) -> Optional[pd.DataFrame]: ...

    def get_raw_ohlcv_from_db(self, market: str, symbol: str) -> Optional[pd.DataFrame]: ...

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame: ...

    def create_basic_lag_features(
        self, df: pd.DataFrame, n_lags: int = 10
    ) -> Tuple[pd.DataFrame, pd.Series]: ...

    def download(
        self,
        ticker: str,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Optional[pd.DataFrame]: ...

    def get_market_regime(self, proxy_df: pd.DataFrame) -> pd.Series: ...


_port: Optional[BacktestDataPort] = None


def set_backtest_data_port(port: BacktestDataPort) -> None:
    """テストや orchestration から実装を注入するためのセッター。"""
    global _port
    _port = port


def get_backtest_data_port() -> BacktestDataPort:
    """デフォルト実装（BacktestMarketDataAdapter）を遅延初期化して返す。"""
    global _port
    if _port is None:
        from src.market_data.backtest_adapter import BacktestMarketDataAdapter

        _port = BacktestMarketDataAdapter()
    return _port
