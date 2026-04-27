"""
ドメイン型定義（移行期間中の後方互換 re-export + 未移行型）

フェーズ1で移動済みの型は各 BC の types.py に定義されています。
このファイルは移行期間中の後方互換 re-export として機能します。

移行状況（フェーズ1完了）:
  - SymbolTask        → src.watchlist.types
  - FeatureLoadResult → src.analysis.types
  - TrainingMetrics   → src.prediction.types
  - PredictionResult  → src.prediction.types

以下は未移行（フェーズ2〜3で順次移動予定）:
  - TradingGateStatus        → trading/types.py（フェーズ2）
  - MarketPredictionSnapshot → reporting/types.py（フェーズ3）
  - WatchlistPredictionRow   → watchlist/types.py（フェーズ3）
  - StressTestResult         → backtest/types.py（フェーズ3）
  - WatchlistPredictionView  → watchlist/types.py（フェーズ3）
  - ShapFeatureContribution  → prediction/types.py（フェーズ3）
  - SignalSnapshot            → prediction/types.py（フェーズ3）
  - SchedulerJobStatus        → orchestration/types.py（フェーズ3）
  - MonthlyReportSummary      → reporting/types.py（フェーズ3）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# analysis BC（共有カーネル）
from src.analysis.types import FeatureLoadResult  # noqa: F401

# prediction BC
from src.prediction.types import PredictionResult, TrainingMetrics  # noqa: F401

# watchlist BC
from src.watchlist.types import SymbolTask  # noqa: F401

# ---------------------------------------------------------------------------
# 未移行型（フェーズ2〜3で各 BC に移動予定）
# ---------------------------------------------------------------------------


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
class StressTestResult:
    """ストレステスト1銘柄・1シナリオの結果。"""

    market: str
    symbol: str
    scenario_name: str  # "corona" / "lehman"
    period_start: str  # "2020-02-01"
    period_end: str  # "2020-03-31"
    mdd: float  # 最大ドローダウン（負の値）
    sharpe_ratio: float
    total_return: float
    win_rate: float
    num_trades: int
    max_consecutive_losses: int
    mdd_pass: bool  # abs(mdd) <= MDD_THRESHOLD (0.15)


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
