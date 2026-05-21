"""
reporting BC の型定義。

Discord / レポート出力で使用するドメイン型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.prediction.types import PredictionResult


@dataclass
class WatchlistPredictionRow:
    """ウォッチリスト表示用の1行。"""

    symbol: str
    current_price: Optional[float]
    avg_pred_price: Optional[float]
    diff_ratio: Optional[float]

    @classmethod
    def to_dataframe(cls, rows: list["WatchlistPredictionRow"]) -> pd.DataFrame:
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
class MarketPredictionSnapshot:
    """Discord などの表示用にまとめた市場別予測スナップショット。"""

    market: str
    top_results: list[PredictionResult] = field(default_factory=list)
    worst_results: list[PredictionResult] = field(default_factory=list)


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
