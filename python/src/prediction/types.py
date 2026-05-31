"""
prediction BC の型定義。

AI モデル学習・予測パイプラインで使用するドメイン型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.domain.types import HorizonResult as HorizonResult  # noqa: F401
from src.domain.types import PredictionResult as PredictionResult  # noqa: F401
from src.domain.types import ShapFeatureContribution as ShapFeatureContribution  # noqa: F401
from src.domain.types import SignalSnapshot as SignalSnapshot  # noqa: F401


@dataclass
class TrainingMetrics:
    """モデル学習後の in-sample 評価指標。

    save_model_metrics() がこの型を受け取る。
    """

    rmse: float
    directional_accuracy: float
    n_samples: int


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


@dataclass
class MarketPredictionSnapshot:
    """Discord などの表示用にまとめた市場別予測スナップショット。"""

    market: str
    top_results: list[PredictionResult] = field(default_factory=list)
    worst_results: list[PredictionResult] = field(default_factory=list)


@dataclass
class DriftMonitorResult:
    """週次 Hit Rate ドリフト検知結果。

    Attributes:
        checked_at: 検査実行日時（ISO 8601）
        current_week: 当週の week_start 文字列（例: "2026-05-11"）
        current_hit_rate: 当週の Hit Rate（方向正解率）
        avg_hit_rate: 過去 alert_weeks 週の平均 Hit Rate
        drop_ratio: (avg - current) / avg — 正の値は低下を示す
        is_drifted: drop_ratio >= alert_threshold のとき True
        alert_weeks: 比較対象の週数
        alert_threshold: ドリフト判定閾値（例: 0.05 = 5%）
    """

    checked_at: str
    current_week: Optional[str]
    current_hit_rate: Optional[float]
    avg_hit_rate: Optional[float]
    drop_ratio: Optional[float]
    is_drifted: bool
    alert_weeks: int
    alert_threshold: float
