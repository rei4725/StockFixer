"""
MarketDataPort - market_data BC への依存を逆転させるポート定義。

prediction BC はこのポートを通じて market_data BC の機能を利用する。
"""

from __future__ import annotations

from typing import Any, List, Optional, Protocol, Tuple, Union, runtime_checkable

import pandas as pd


@runtime_checkable
class MarketDataPort(Protocol):
    """prediction 用 market_data アクセスポートのインターフェース。"""

    def get_stock_data(self, market: str, symbol: str, start: Any, end: Any) -> pd.DataFrame: ...

    def fetch_cross_asset_features(self, start: str, end: str) -> Optional[pd.DataFrame]: ...

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
    """注入済みの MarketDataPort を返す。未注入なら RuntimeError。

    注入は orchestration の合成ルートで行う:
    `src.orchestration.port_wiring.wire_ports()`（エントリポイント起動時に呼ぶ）。
    テストや個別注入は `set_market_data_port()` を使う。
    """
    if _port is None:
        raise RuntimeError(
            "MarketDataPort が未注入です。エントリポイントで "
            "src.orchestration.port_wiring.wire_ports() を呼ぶか、"
            "set_market_data_port() で実装を注入してください。"
        )
    return _port
