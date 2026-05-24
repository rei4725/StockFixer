"""rule_engine BC のポート定義

prediction / market_data への依存をプロトコルで抽象化する。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class TradingRule(Protocol):
    """ルールベースシグナル生成プロトコル"""

    name: str
    description: str

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """OHLCV + テクニカル指標 DataFrame からシグナル（1=buy/-1=sell/0=hold）を生成する"""
        ...
