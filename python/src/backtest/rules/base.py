"""
トレーディングルールの基底プロトコル

各ルールはこのプロトコルを実装し、OHLCV+テクニカル指標 DataFrame から
シグナル Series (1=buy, -1=sell, 0=hold) を生成する。
"""

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

        Args:
            df: Open/High/Low/Close/Volume + テクニカル指標列を含む DataFrame
                （add_technical_indicators 適用済みを想定）

        Returns:
            pd.Series: 1=buy, -1=sell, 0=hold（インデックスは df と同じ）
        """
        ...
