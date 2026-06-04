"""
BacktestDataPort - market_data BC への依存を逆転させるポート定義。

backtest BC はこのポートを通じて market_data BC の機能を利用する。
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Tuple, runtime_checkable

import pandas as pd


@runtime_checkable
class BacktestDataPort(Protocol):
    """バックテスト用 market_data アクセスポートのインターフェース。"""

    def get_stock_data(
        self, market: str, ticker: str, start: str, end: str
    ) -> Optional[pd.DataFrame]: ...

    def get_raw_ohlcv_from_db(
        self, market: str, symbol: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> Optional[pd.DataFrame]: ...

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame: ...

    def create_basic_lag_features(
        self, df: pd.DataFrame, n_lags: int = 10
    ) -> Tuple[pd.DataFrame, pd.Series]: ...

    def classify_regime(self, df: pd.DataFrame, **kwargs: Any) -> pd.Series: ...

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
    """注入済みの BacktestDataPort を返す。未注入なら RuntimeError。

    注入は orchestration の合成ルートで行う:
    `src.orchestration.port_wiring.wire_ports()`（エントリポイント起動時に呼ぶ）。
    テストや個別注入は `set_backtest_data_port()` を使う。
    """
    if _port is None:
        raise RuntimeError(
            "BacktestDataPort が未注入です。エントリポイントで "
            "src.orchestration.port_wiring.wire_ports() を呼ぶか、"
            "set_backtest_data_port() で実装を注入してください。"
        )
    return _port
