"""
月次KPIレポートパイプライン（R-203）

最重要KPI（Net Return / Max Drawdown / Sharpe Ratio）と補助KPI
（Hit Rate / Avg Slippage）を集計し MonthlyReportSummary を返す。

データソース:
    - Net Return / Sharpe / Max Drawdown : 最新の Walk-Forward レポート CSV
    - Hit Rate                           : prediction_accuracy テーブル（直近30日）
    - Avg Slippage                       : paper_real_diff テーブル（直近30日）
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.reporting.types import MonthlyReportSummary
from src.utils.data_path_utils import get_results_dir
from src.utils.db import load_paper_real_diff_summary, load_prediction_accuracy
from src.utils.logger import get_logger

logger = get_logger(__name__)

_REPORT_DAYS = 30  # Hit Rate / Avg Slippage の集計期間（日）


# ---------------------------------------------------------------------------
# 内部ヘルパー
# ---------------------------------------------------------------------------


def _load_latest_wf_summary() -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """最新の Walk-Forward サマリー CSV を返す。(DataFrame, ファイル名)"""
    report_dir = Path(get_results_dir()) / "backtest" / "walk_forward_reports"
    candidates = sorted(report_dir.glob("wf_summary_*.csv"), reverse=True)
    if not candidates:
        logger.warning("Walk-Forward レポート CSV が見つかりません")
        return None, None
    path = candidates[0]
    try:
        df = pd.read_csv(path)
        logger.info(f"WF サマリー読み込み: {path.name} ({len(df)} 銘柄)")
        return df, path.name
    except Exception as e:
        logger.error(f"WF サマリー読み込み失敗: {e}", exc_info=True)
        return None, None


def _mean_metric(df: pd.DataFrame, col: str) -> Optional[float]:
    if col not in df.columns:
        return None
    val = pd.to_numeric(df[col], errors="coerce").mean()
    return float(val) if not pd.isna(val) else None


def _compute_hit_rate(days: int = _REPORT_DAYS) -> Optional[float]:
    """prediction_accuracy テーブルから方向一致率を集計する。"""
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
    """paper_real_diff テーブルから平均スリッページを取得する。"""
    try:
        summary = load_paper_real_diff_summary(recent_days=days)
        val = summary.get("avg_paper_slippage")
        return float(val) if val is not None else None
    except Exception as e:
        logger.error(f"avg_slippage 取得失敗: {e}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# パブリック API
# ---------------------------------------------------------------------------


def run_monthly_report(target_month: Optional[str] = None) -> MonthlyReportSummary:
    """
    月次KPIサマリーを集計して MonthlyReportSummary を返す。

    Args:
        target_month: 対象年月 "YYYY-MM"（省略時は当月）

    Returns:
        MonthlyReportSummary
    """
    if target_month is None:
        target_month = datetime.now().strftime("%Y-%m")

    generated_at = datetime.now().isoformat()

    # ---- Walk-Forward KPI ----
    wf_df, wf_file = _load_latest_wf_summary()
    net_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    symbol_count: Optional[int] = None

    if wf_df is not None and not wf_df.empty:
        net_return = _mean_metric(wf_df, "total_return")
        max_drawdown = _mean_metric(wf_df, "max_drawdown")
        sharpe_ratio = _mean_metric(wf_df, "sharpe_ratio")
        symbol_count = len(wf_df)

    # ---- 補助KPI ----
    hit_rate = _compute_hit_rate()
    avg_slippage = _compute_avg_slippage()

    summary = MonthlyReportSummary(
        generated_at=generated_at,
        target_month=target_month,
        net_return=net_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        hit_rate=hit_rate,
        avg_slippage=avg_slippage,
        wf_snapshot_file=wf_file,
        symbol_count=symbol_count,
    )
    logger.info(
        f"月次レポート生成: {target_month} "
        f"NetReturn={net_return} Sharpe={sharpe_ratio} MDD={max_drawdown} "
        f"HitRate={hit_rate} AvgSlippage={avg_slippage}"
    )
    return summary
