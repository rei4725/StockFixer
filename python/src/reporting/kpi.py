"""
reporting BC の月次 KPI 集約サービス。

reporting BC が予測精度データを参照する際の唯一の窓口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.utils.db import load_drift_summary, load_paper_real_diff_summary, load_prediction_accuracy
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


def get_monthly_kpis(days: int = _REPORT_DAYS) -> MonthlyKPI:
    """hit_rate / avg_slippage / drift_count を集約して返す。"""
    return MonthlyKPI(
        hit_rate=_compute_hit_rate(days),
        avg_slippage=_compute_avg_slippage(days),
        drift_count=_compute_drift_count(days),
        diff_summary=_load_diff_summary(days),
    )


def _compute_hit_rate(days: int = _REPORT_DAYS) -> Optional[float]:
    df = load_prediction_accuracy(horizon=1, limit=5000)
    if df.empty or "direction_match" not in df.columns:
        return None
    if "checked_at" in df.columns:
        df["checked_at"] = pd.to_datetime(df["checked_at"], errors="coerce")
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df["checked_at"] >= cutoff]
    if df.empty:
        return None
    return float(df["direction_match"].astype(float).mean())


def _compute_avg_slippage(days: int = _REPORT_DAYS) -> Optional[float]:
    try:
        summary = load_paper_real_diff_summary(recent_days=days)
        val = summary.get("avg_paper_slippage")
        return float(val) if val is not None else None
    except Exception as e:
        logger.error(f"avg_slippage 取得失敗: {e}", exc_info=True)
        return None


def _load_diff_summary(days: int = _REPORT_DAYS) -> dict:
    try:
        return load_paper_real_diff_summary(recent_days=days)
    except Exception as e:
        logger.error(f"diff_summary 取得失敗: {e}", exc_info=True)
        return dict(_EMPTY_DIFF)


def _compute_drift_count(days: int = _REPORT_DAYS) -> int:
    drift_df = load_drift_summary(horizon=1, recent_n=days)
    if drift_df is None or drift_df.empty:
        return 0
    return int(
        (
            (drift_df.get("mean_abs_error", pd.Series(dtype=float)) >= 0.02)
            | (drift_df.get("direction_accuracy", pd.Series(dtype=float)) <= 0.45)
        ).sum()
    )
