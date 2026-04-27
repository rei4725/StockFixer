"""
prediction BC の型定義。

AI モデル学習・予測パイプラインで使用するドメイン型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class TrainingMetrics:
    """モデル学習後の in-sample 評価指標。

    save_model_metrics() がこの型を受け取る。
    """

    rmse: float
    directional_accuracy: float
    n_samples: int


@dataclass
class PredictionResult:
    """1銘柄の予測結果。

    predict_single_stock / predict_with_unified_model の戻り値として使い、
    prediction_pipeline → DB保存 → Discord出力まで全層を型安全に繋ぐ。
    """

    market: str
    symbol: str
    current_price: float
    avg_pred_price: float
    diff_ratio: float
    model_count: int

    # 多ホライズン（省略可）
    avg_pred_price_3d: Optional[float] = None
    avg_pred_price_5d: Optional[float] = None
    avg_pred_price_10d: Optional[float] = None
    diff_ratio_3d: Optional[float] = None
    diff_ratio_5d: Optional[float] = None
    diff_ratio_10d: Optional[float] = None
    confluence_score: Optional[int] = None
    confidence_ratio: Optional[float] = None  # 1/(1+model_std); 1.0=最大信頼度（モデル間分散が小さい）

    # A/B テスト（シャドーモード）
    model_version: Optional[str] = None  # "production" / "challenger" / 任意バージョン文字列

    # ------------------------------------------------------------------
    # 変換メソッド（変換知識はここに1箇所）
    # ------------------------------------------------------------------

    @classmethod
    def to_dataframe(cls, results: list[PredictionResult]) -> pd.DataFrame:
        """PredictionResult のリストを保存・集計用 DataFrame に変換する。"""
        rows = []
        for r in results:
            row: dict = {
                "market": r.market,
                "symbol": r.symbol,
                "current_price": r.current_price,
                "avg_pred_price": r.avg_pred_price,
                "diff_ratio": r.diff_ratio,
                "model_count": r.model_count,
            }
            if r.avg_pred_price_3d is not None:
                row["avg_pred_price_3d"] = r.avg_pred_price_3d
            if r.avg_pred_price_5d is not None:
                row["avg_pred_price_5d"] = r.avg_pred_price_5d
            if r.avg_pred_price_10d is not None:
                row["avg_pred_price_10d"] = r.avg_pred_price_10d
            if r.diff_ratio_3d is not None:
                row["diff_ratio_3d"] = r.diff_ratio_3d
            if r.diff_ratio_5d is not None:
                row["diff_ratio_5d"] = r.diff_ratio_5d
            if r.diff_ratio_10d is not None:
                row["diff_ratio_10d"] = r.diff_ratio_10d
            if r.confluence_score is not None:
                row["confluence_score"] = r.confluence_score
            if r.confidence_ratio is not None:
                row["confidence_ratio"] = r.confidence_ratio
            if r.model_version is not None:
                row["model_version"] = r.model_version
            rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    @classmethod
    def from_dataframe_row(cls, row: pd.Series) -> PredictionResult:
        """DataFrame の1行から PredictionResult を復元する（DB読み込み時に使用）。"""

        def _opt_float(key: str) -> Optional[float]:
            val = row.get(key)
            return float(val) if val is not None and not pd.isna(val) else None

        def _opt_int(key: str) -> Optional[int]:
            val = row.get(key)
            return int(val) if val is not None and not pd.isna(val) else None

        def _opt_str(key: str) -> Optional[str]:
            val = row.get(key)
            return str(val) if val is not None and not pd.isna(val) else None

        return cls(
            market=str(row["market"]),
            symbol=str(row["symbol"]),
            current_price=float(row["current_price"]),
            avg_pred_price=float(row["avg_pred_price"]),
            diff_ratio=float(row["diff_ratio"]),
            model_count=int(row["model_count"]),
            avg_pred_price_3d=_opt_float("avg_pred_price_3d"),
            avg_pred_price_5d=_opt_float("avg_pred_price_5d"),
            avg_pred_price_10d=_opt_float("avg_pred_price_10d"),
            diff_ratio_3d=_opt_float("diff_ratio_3d"),
            diff_ratio_5d=_opt_float("diff_ratio_5d"),
            diff_ratio_10d=_opt_float("diff_ratio_10d"),
            confluence_score=_opt_int("confluence_score"),
            confidence_ratio=_opt_float("confidence_ratio"),
            model_version=_opt_str("model_version"),
        )
