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


@dataclass
class TrainingMetrics:
    """モデル学習後の in-sample 評価指標。

    save_model_metrics() がこの型を受け取る。
    """

    rmse: float
    directional_accuracy: float
    n_samples: int


@dataclass
class TradingGateStatus:
    """発注前リスクゲートの評価結果。"""

    is_allowed: bool
    stop_active: bool
    reason_code: Optional[str] = None
    reason: Optional[str] = None
    daily_loss: float = 0.0
    daily_loss_limit: Optional[float] = None
    consecutive_losses: int = 0
    consecutive_loss_limit: Optional[int] = None
    position_count: int = 0
    max_positions: Optional[int] = None


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
    confidence_ratio: Optional[float] = None  # 1/(1+model_std); 1.0=最大信頼度（モデル間分散が小さい）

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
            confidence_ratio=_opt_float("confidence_ratio"),
        )


@dataclass
class MarketPredictionSnapshot:
    """Discord などの表示用にまとめた市場別予測スナップショット。"""

    market: str
    top_results: list[PredictionResult] = field(default_factory=list)
    worst_results: list[PredictionResult] = field(default_factory=list)


@dataclass
class WatchlistPredictionRow:
    """ウォッチリスト表示用の1行。"""

    symbol: str
    current_price: Optional[float]
    avg_pred_price: Optional[float]
    diff_ratio: Optional[float]

    @classmethod
    def to_dataframe(cls, rows: list[WatchlistPredictionRow]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "symbol": row.symbol,
                    "current_price": row.current_price,
                    "avg_pred_price": row.avg_pred_price,
                    "diff_ratio": row.diff_ratio,
                }
                for row in rows
            ]
        )


@dataclass
class WatchlistPredictionView:
    """ウォッチリスト表示の取得結果。"""

    rows: list[WatchlistPredictionRow] = field(default_factory=list)
    error_message: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.error_message is None


@dataclass
class ShapFeatureContribution:
    """SHAP 説明の1特徴量。"""

    feature: str
    shap_value: float


@dataclass
class SignalSnapshot:
    """Discord signal コマンド向けの予測スナップショット。"""

    prediction: PredictionResult
    shap_direction: Optional[str] = None
    top_features: list[ShapFeatureContribution] = field(default_factory=list)


@dataclass
class SchedulerJobStatus:
    """スケジューラ状態表示用のジョブステータス。"""

    job_id: str
    label: str
    last_run_at: Optional[str]
    status: str


# ---------------------------------------------------------------------------
# 月次レポート型
# ---------------------------------------------------------------------------


@dataclass
class MonthlyReportSummary:
    """月次KPIサマリー。

    monthly_report_pipeline.run_monthly_report() が返す。
    Discord /monthlyreport コマンドや run_monthly_report.py から参照する。
    """

    generated_at: str  # ISO形式 datetime
    target_month: str  # "YYYY-MM"

    # 最重要KPI
    net_return: Optional[float]  # 手数料考慮後リターン（WF fold平均）
    max_drawdown: Optional[float]  # 最大ドローダウン（WF fold平均）
    sharpe_ratio: Optional[float]  # シャープレシオ（WF fold平均）

    # 補助KPI
    hit_rate: Optional[float]  # 方向一致率（直近30日、prediction_accuracy）
    avg_slippage: Optional[float]  # papperスリッページ平均（直近30日）

    # メタ情報
    wf_snapshot_file: Optional[str] = None  # 参照したWFレポートCSVファイル名
    symbol_count: Optional[int] = None  # 集計対象銘柄数
