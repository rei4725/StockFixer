"""
ドメイン型定義

パイプライン全体で共有される構造化データクラスを一元管理する。
dict / 生 pd.DataFrame の代わりに使用することで、
同じ構造の変更が全層に自動伝播する設計にする。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Task 型
# ---------------------------------------------------------------------------


@dataclass
class SymbolTask:
    """バッチ処理の単位タスク（market + symbol + horizon）。

    batch_runner.load_target_symbols() / run_parallel() で使用する。
    """

    market: str
    symbol: str
    horizon: int = 1


# ---------------------------------------------------------------------------
# 学習パイプライン型
# ---------------------------------------------------------------------------


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

    # batch_runner.print_summary が "status" / "market" / "symbol" キーに依存するため
    # dict 互換のアクセスを提供する
    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


@dataclass
class TrainingMetrics:
    """モデル学習後の in-sample 評価指標。

    save_model_metrics() がこの型を受け取る。
    """

    rmse: float
    directional_accuracy: float
    n_samples: int


# ---------------------------------------------------------------------------
# 予測結果型
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """1銘柄の予測結果。

    predict_single_stock / predict_with_unified_model の戻り値として使い、
    prediction_pipeline → DB保存 → Discord出力まで全层を型安全に繋ぐ。
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
        )
