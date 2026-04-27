"""
analysis BC の型定義（共有カーネル）。

テクニカル分析・特徴量生成パイプラインで使用するドメイン型。
prediction / backtest / strategy BC から参照される共有カーネル型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class FeatureLoadResult:
    """load_features_for_training の戻り値。

    status:
        "success" — X / y 付き
        "skip"    — データ不足（reason に理由）
        "error"   — 例外発生（error に文字列）
    """

    status: str
    market: str
    symbol: str
    X: Optional[pd.DataFrame] = field(default=None, repr=False)
    y: Optional[pd.Series] = field(default=None, repr=False)
    reason: Optional[str] = None
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.status == "success" and self.X is not None
