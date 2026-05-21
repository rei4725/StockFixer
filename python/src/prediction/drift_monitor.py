"""
週次 Hit Rate トレンド監視・モデルドリフト検知（R-274）

prediction_accuracy テーブルの Hit Rate を週次集計し、
過去 N 週平均から閾値以上低下したらドリフトとして検知する。

設定: config/settings.py の DRIFT_ALERT_WEEKS / DRIFT_ALERT_THRESHOLD
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from src.prediction.types import DriftMonitorResult  # noqa: F401
from src.utils.db import load_weekly_accuracy_snapshots
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_weekly_hit_rates_from_snapshots(n_weeks: int) -> pd.DataFrame:
    """accuracy_weekly_snapshots から週次 Hit Rate を取得する。

    Returns:
        DataFrame [week_start, hit_rate] (week_start 降順)。空の場合は空 DataFrame。
    """
    snap = load_weekly_accuracy_snapshots(n_weeks=n_weeks)
    if snap.empty or "direction_accuracy" not in snap.columns:
        return pd.DataFrame()

    weekly = (
        snap.groupby("week_start")["direction_accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"direction_accuracy": "hit_rate"})
        .sort_values("week_start", ascending=False)
        .reset_index(drop=True)
    )
    return weekly


def _load_weekly_hit_rates_direct(n_weeks: int, horizon: int) -> pd.DataFrame:
    """prediction_accuracy テーブルを直接集計し週次 Hit Rate を返す（フォールバック）。

    Returns:
        DataFrame [week_start, hit_rate] (week_start 降順)。空の場合は空 DataFrame。
    """
    with _db_connection() as con:
        try:
            df = con.execute(
                f"""
                SELECT
                    date_trunc('week', checked_at) AS week_start,
                    AVG(CAST(direction_match AS INTEGER)) AS hit_rate
                FROM prediction_accuracy
                WHERE horizon = ?
                  AND direction_match IS NOT NULL
                  AND checked_at IS NOT NULL
                GROUP BY date_trunc('week', checked_at)
                ORDER BY week_start DESC
                LIMIT {int(n_weeks)}
                """,
                [horizon],
            ).fetchdf()
            if df.empty:
                return pd.DataFrame()
            df["week_start"] = df["week_start"].astype(str)
            return df[["week_start", "hit_rate"]].reset_index(drop=True)
        except Exception as e:
            logger.error("週次 Hit Rate 直接集計失敗: %s", e, exc_info=True)
            return pd.DataFrame()


def _load_weekly_hit_rates(n_weeks: int, horizon: int) -> pd.DataFrame:
    """週次 Hit Rate を返す。スナップショット優先、不足時は直接集計。

    Args:
        n_weeks: 取得する週数
        horizon: 対象ホライズン（直接集計フォールバック用）

    Returns:
        DataFrame [week_start, hit_rate] (week_start 降順)
    """
    weekly = _load_weekly_hit_rates_from_snapshots(n_weeks)
    if len(weekly) >= 2:
        return weekly

    logger.info(
        "accuracy_weekly_snapshots データ不足 (%d 週) — prediction_accuracy を直接集計します",
        len(weekly),
    )
    return _load_weekly_hit_rates_direct(n_weeks, horizon)


def check_weekly_hit_rate_drift(
    weeks: int = 4,
    threshold: float = 0.05,
    horizon: int = 1,
) -> DriftMonitorResult:
    """週次 Hit Rate トレンドを監視し、モデルドリフトを検知する。

    過去 weeks 週の平均 Hit Rate と当週の Hit Rate を比較し、
    (avg - current) / avg >= threshold の場合に is_drifted=True を返す。

    Args:
        weeks: 比較対象の週数（デフォルト 4）
        threshold: ドリフト判定の低下率閾値（デフォルト 0.05 = 5%）
        horizon: 対象ホライズン（デフォルト 1）

    Returns:
        DriftMonitorResult
    """
    checked_at = datetime.now().isoformat()

    weekly = _load_weekly_hit_rates(n_weeks=weeks + 1, horizon=horizon)

    if weekly.empty or len(weekly) < 2:
        logger.warning("週次 Hit Rate データ不足: %d 週分 (最低 2 週必要)", len(weekly))
        return DriftMonitorResult(
            checked_at=checked_at,
            current_week=None,
            current_hit_rate=None,
            avg_hit_rate=None,
            drop_ratio=None,
            is_drifted=False,
            alert_weeks=weeks,
            alert_threshold=threshold,
        )

    weekly = weekly.sort_values("week_start", ascending=False).reset_index(drop=True)
    current_week = str(weekly.iloc[0]["week_start"])
    current_hit_rate = float(weekly.iloc[0]["hit_rate"])

    comparison = weekly.iloc[1 : weeks + 1]
    avg_hit_rate = float(comparison["hit_rate"].mean())

    if avg_hit_rate > 0:
        drop_ratio = (avg_hit_rate - current_hit_rate) / avg_hit_rate
    else:
        drop_ratio = 0.0

    is_drifted = drop_ratio >= threshold

    logger.info(
        "週次 Hit Rate ドリフト検査: week=%s current=%.3f avg=%.3f drop=%.2f%% drifted=%s",
        current_week,
        current_hit_rate,
        avg_hit_rate,
        drop_ratio * 100,
        is_drifted,
    )

    return DriftMonitorResult(
        checked_at=checked_at,
        current_week=current_week,
        current_hit_rate=current_hit_rate,
        avg_hit_rate=avg_hit_rate,
        drop_ratio=drop_ratio,
        is_drifted=is_drifted,
        alert_weeks=weeks,
        alert_threshold=threshold,
    )
