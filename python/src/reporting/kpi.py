"""
reporting BC の月次 KPI 集約サービス。

データは入口（monthly.py）が AnalyticsQuery から取得して渡す。本モジュールは
DB を読まず、受け取ったデータを集約するだけの内側の関数群である。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_REPORT_DAYS = 30

_EMPTY_DIFF: dict = {
    "tracked_count": 0,
    "comparable_count": 0,
    "avg_paper_slippage": 0.0,
    "avg_real_slippage": 0.0,
    "avg_abs_price_diff": 0.0,
    "avg_abs_diff_ratio": 0.0,
    "max_abs_price_diff": 0.0,
}


@dataclass
class MonthlyKPI:
    """月次レポート用 KPI 集約結果。"""

    hit_rate: Optional[float]
    avg_slippage: Optional[float]
    drift_count: int
    diff_summary: dict = field(default_factory=dict)


def get_monthly_kpis(
    days: int = _REPORT_DAYS,
    *,
    diff_summary: Optional[dict],
    accuracy_df: pd.DataFrame,
    drift_df: pd.DataFrame,
) -> MonthlyKPI:
    """hit_rate / avg_slippage / drift_count を集約して返す。

    入力はすべて入口が AnalyticsQuery から取得して渡す（自分では読みに行かない）。
    accuracy_df は prediction_accuracy(horizon=1, limit=5000)、
    drift_df は drift_summary(horizon=1, recent_n=days) の戻り値を想定する。

    diff_summary の取得に失敗した場合は None を渡すこと。None のときは avg_slippage を
    「計測されたゼロ」(0.0) ではなく「データなし」(None) として扱い、
    diff_summary フィールドには表示用の _EMPTY_DIFF を補う。
    """
    if diff_summary is None:
        avg_slippage = None
        resolved_diff_summary = dict(_EMPTY_DIFF)
    else:
        raw_avg_slippage = diff_summary.get("avg_paper_slippage")
        avg_slippage = float(raw_avg_slippage) if raw_avg_slippage is not None else None
        resolved_diff_summary = diff_summary
    return MonthlyKPI(
        hit_rate=_compute_hit_rate(accuracy_df, days),
        avg_slippage=avg_slippage,
        drift_count=_compute_drift_count(drift_df),
        diff_summary=resolved_diff_summary,
    )


def _compute_hit_rate(accuracy_df: pd.DataFrame, days: int = _REPORT_DAYS) -> Optional[float]:
    # 呼び出し元の DataFrame を書き換えないようコピーしてから列を変換する
    df = accuracy_df.copy()
    if df.empty or "direction_match" not in df.columns:
        return None
    if "checked_at" in df.columns:
        df["checked_at"] = pd.to_datetime(df["checked_at"], errors="coerce")
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df["checked_at"] >= cutoff]
    if df.empty:
        return None
    return float(df["direction_match"].astype(float).mean())


def _compute_drift_count(drift_df: pd.DataFrame) -> int:
    if drift_df is None or drift_df.empty:
        return 0
    return int(
        (
            (drift_df.get("mean_abs_error", pd.Series(dtype=float)) >= 0.02)
            | (drift_df.get("direction_accuracy", pd.Series(dtype=float)) <= 0.45)
        ).sum()
    )
