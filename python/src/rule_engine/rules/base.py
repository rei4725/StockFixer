"""トレーディングルールの基底プロトコル"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class TradingRule(Protocol):
    name: str
    description: str

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        OHLCV + テクニカル指標 DataFrame からシグナルを生成する。

        Returns:
            pd.Series: 1=buy, -1=sell, 0=hold（インデックスは df と同じ）
        """
        ...
