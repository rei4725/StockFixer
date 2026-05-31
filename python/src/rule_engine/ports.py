"""rule_engine BC のポート定義

prediction / market_data への依存をプロトコルで抽象化する。
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class TradingRule(Protocol):
    """ルールベースシグナル生成プロトコル"""

    name: str
    description: str

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """OHLCV + テクニカル指標 DataFrame からシグナル（1=buy/-1=sell/0=hold）を生成する"""
        ...


@runtime_checkable
class OHLCVWithIndicatorsPort(Protocol):
    """OHLCV + テクニカル指標付きデータ取得プロトコル"""

    def get_ohlcv_with_indicators(self, ticker: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """指定ティッカーの OHLCV + テクニカル指標付き DataFrame を返す。

        データなしまたはエラー時は None を返す。
        """
        ...
