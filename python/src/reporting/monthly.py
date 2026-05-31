"""
月次KPIレポートパイプライン（R-203）

最重要KPI（Net Return / Max Drawdown / Sharpe Ratio）と補助KPI
（Hit Rate / Avg Slippage）を集計し MonthlyReportSummary を返す。

データソース:
    - Net Return / Sharpe / Max Drawdown : 最新の Walk-Forward レポート CSV
    - Hit Rate / Avg Slippage / Drift    : prediction.kpi_service 経由
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from src.reporting.kpi import get_monthly_kpis
from src.reporting.types import MonthlyReportSummary
from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
    kpi = get_monthly_kpis()
    hit_rate = kpi.hit_rate
    avg_slippage = kpi.avg_slippage

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


def save_monthly_report_to_file(
    summary: MonthlyReportSummary,
    drift_checker: Optional[Callable[[], Any]] = None,
) -> str:
    """
    月次KPIサマリーを Markdown ファイルとして保存する（R-203）。

    保存先: results/monthly/YYYY-MM_report.md

    Args:
        summary: run_monthly_report() が返す MonthlyReportSummary
        drift_checker: 週次 Hit Rate ドリフト検査を実行するコールバック（省略可）

    Returns:
        保存先ファイルパス
    """
    output_dir = Path(get_results_dir()) / "monthly"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{summary.target_month}_report.md"

    def _pct(val: Optional[float]) -> str:
        return f"{val * 100:.2f}%" if val is not None else "N/A"

    def _f2(val: Optional[float]) -> str:
        return f"{val:.2f}" if val is not None else "N/A"

    # paper/real 乖離サマリー・ドリフト集計
    _kpi = get_monthly_kpis()
    diff = _kpi.diff_summary
    drift_count = _kpi.drift_count

    # 週次 Hit Rate ドリフト
    if drift_checker is not None:
        try:
            drift_result = drift_checker()
        except Exception as e:
            logger.error("週次 Hit Rate ドリフト検査失敗: %s", e, exc_info=True)
            drift_result = None
    else:
        drift_result = None

    def _drift_row(label: str, val) -> str:
        return f"| {label} | {val} |"

    weekly_lines = ["## 週次 Hit Rate トレンド", ""]
    if drift_result is not None and drift_result.current_week is not None:
        weekly_lines += [
            "| 項目 | 値 |",
            "|---|---|",
            _drift_row("当週", drift_result.current_week),
            _drift_row("当週 Hit Rate", _pct(drift_result.current_hit_rate)),
            _drift_row(
                f"過去 {drift_result.alert_weeks} 週平均 Hit Rate",
                _pct(drift_result.avg_hit_rate),
            ),
            _drift_row("低下率", _pct(drift_result.drop_ratio)),
            _drift_row("ドリフト検知", "**YES ⚠️**" if drift_result.is_drifted else "なし"),
            _drift_row("アラート閾値", _pct(drift_result.alert_threshold)),
            "",
        ]
    else:
        weekly_lines += ["週次スナップショットデータなし（2 週分以上必要）", ""]

    lines = [
        f"# StockFixer 月次レポート {summary.target_month}",
        "",
        f"生成日時: {summary.generated_at}",
        "",
        "## KPI サマリー",
        "",
        "| 指標 | 値 |",
        "|---|---|",
        f"| Net Return | {_pct(summary.net_return)} |",
        f"| Max Drawdown | {_pct(summary.max_drawdown)} |",
        f"| Sharpe Ratio | {_f2(summary.sharpe_ratio)} |",
        f"| Hit Rate | {_pct(summary.hit_rate)} |",
        f"| Avg Slippage | {_pct(summary.avg_slippage)} |",
        f"| 集計銘柄数 | {summary.symbol_count or 'N/A'} |",
        f"| WF スナップショット | {summary.wf_snapshot_file or 'N/A'} |",
        "",
        *weekly_lines,
        "## paper/real 乖離（直近30日）",
        "",
        "| 指標 | 値 |",
        "|---|---|",
        f"| 追跡件数 | {diff['tracked_count']} |",
        f"| 比較可能件数 | {diff['comparable_count']} |",
        f"| 平均price乖離 | ¥{diff['avg_abs_price_diff']:.2f} |",
        f"| 平均乖離率 | {diff['avg_abs_diff_ratio'] * 100:.4f}% |",
        f"| Paper Slippage | {_pct(diff['avg_paper_slippage'])} |",
        f"| Real Slippage | {_pct(diff['avg_real_slippage'])} |",
        "",
        "## ドリフト状況（直近30日）",
        "",
        f"閾値超過銘柄数: {drift_count}",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("月次レポートを保存: %s", output_path)
    return str(output_path)
