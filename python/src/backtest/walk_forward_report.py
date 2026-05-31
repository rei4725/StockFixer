"""
Walk-Forward 比較レポートパイプライン

標準条件で全銘柄の Walk-Forward 検証を実行し、
前回スナップショットとの差分比較レポートを保存する。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.backtest.pipeline import run_backtest_walk_forward
from src.utils.data_path_utils import ensure_dir, get_results_dir
from src.utils.db.experiment import generate_run_id, save_experiment_run
from src.utils.logger import get_logger

logger = get_logger(__name__)

_REPORT_METRICS = [
    "total_return",
    "sharpe_ratio",
    "max_drawdown",
    "gross_total_return",
    "gross_sharpe_ratio",
    "gross_max_drawdown",
    "cost_impact_return",
    "cost_impact_cash",
    "win_rate",
    "profit_factor",
    "num_trades",
    "short_return",
    "short_num_trades",
]


def _numeric_mean(df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    available = [c for c in columns if c in df.columns]
    if not available:
        return {}
    work = df[available].copy()
    for col in work.columns:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    return {str(k): v for k, v in work.mean().to_dict().items()}


def _summarize_wf_result(wf_df: pd.DataFrame, market: str, symbol: str) -> dict[str, Any]:
    summary = _numeric_mean(wf_df, _REPORT_METRICS)
    summary["market"] = market
    summary["symbol"] = symbol
    summary["fold_count"] = int(len(wf_df))
    return summary


def _find_latest_summary_csv(report_dir: Path, exclude: Optional[Path] = None) -> Optional[Path]:
    candidates = sorted(report_dir.glob("wf_summary_*.csv"), reverse=True)
    for path in candidates:
        if exclude is not None and path.resolve() == exclude.resolve():
            continue
        return path
    return None


def _build_comparison(current_df: pd.DataFrame, prev_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    current = current_df.copy()
    if prev_df is None or prev_df.empty:
        # 初回実行時は current のみ返す
        current["has_previous"] = False
        return current

    merged = current.merge(
        prev_df,
        on=["market", "symbol"],
        how="left",
        suffixes=("_current", "_prev"),
    )

    delta_metrics = [
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "gross_total_return",
        "gross_sharpe_ratio",
        "gross_max_drawdown",
        "cost_impact_return",
        "cost_impact_cash",
        "win_rate",
        "profit_factor",
        "num_trades",
        "short_return",
        "short_num_trades",
    ]
    for metric in delta_metrics:
        cur = f"{metric}_current"
        prv = f"{metric}_prev"
        if cur in merged.columns and prv in merged.columns:
            merged[f"delta_{metric}"] = pd.to_numeric(merged[cur], errors="coerce") - pd.to_numeric(
                merged[prv], errors="coerce"
            )

    col = merged.get("total_return_prev")
    merged["has_previous"] = col.notna() if col is not None else False
    return merged


def _to_markdown_summary(comparison_df: pd.DataFrame, previous_path: Optional[Path]) -> str:
    lines: list[str] = []
    lines.append("# Walk-Forward 比較レポート")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now().isoformat()}")
    lines.append(f"- symbols: {len(comparison_df)}")
    lines.append(f"- previous_snapshot: {previous_path.name if previous_path else 'none'}")
    lines.append("")

    if comparison_df.empty:
        lines.append("データがありません。")
        return "\n".join(lines)

    if "delta_total_return" in comparison_df.columns:
        valid = comparison_df[comparison_df["has_previous"] == True].copy()  # noqa: E712
        if not valid.empty:
            top = valid.sort_values("delta_total_return", ascending=False).head(10)
            bottom = valid.sort_values("delta_total_return", ascending=True).head(10)

            top_cols = [
                "market",
                "symbol",
                "total_return_current",
                "total_return_prev",
                "delta_total_return",
            ]
            bottom_cols = top_cols

            lines.append("## Total Return 改善 Top 10")
            lines.append("")
            lines.append(top[top_cols].to_string(index=False))
            lines.append("")
            lines.append("## Total Return 悪化 Top 10")
            lines.append("")
            lines.append(bottom[bottom_cols].to_string(index=False))
            lines.append("")

    overall_current_cols = [
        c
        for c in ["total_return", "sharpe_ratio", "max_drawdown", "cost_impact_return"]
        if c in comparison_df.columns
    ]
    if overall_current_cols:
        overall = _numeric_mean(comparison_df, overall_current_cols)
        lines.append("## 全体平均（current）")
        lines.append("")
        for key, value in overall.items():
            lines.append(f"- {key}: {value:.6f}")
        lines.append("")

    # ロング単独 vs ロング+ショートの比較セクション
    short_return_col = next(
        (c for c in ["short_return_current", "short_return"] if c in comparison_df.columns),
        None,
    )
    if short_return_col is not None:
        short_df = comparison_df.copy()
        short_return_series = pd.to_numeric(short_df[short_return_col], errors="coerce").fillna(0)
        has_short = short_return_series.abs().sum() > 0

        if has_short:
            long_only_return_col = next(
                (c for c in ["total_return_current", "total_return"] if c in short_df.columns),
                None,
            )
            long_sharpe_col = next(
                (c for c in ["sharpe_ratio_current", "sharpe_ratio"] if c in short_df.columns),
                None,
            )
            long_mdd_col = next(
                (c for c in ["max_drawdown_current", "max_drawdown"] if c in short_df.columns),
                None,
            )

            long_return = (
                pd.to_numeric(short_df[long_only_return_col], errors="coerce").mean()
                if long_only_return_col
                else 0.0
            )
            long_sharpe = (
                pd.to_numeric(short_df[long_sharpe_col], errors="coerce").mean()
                if long_sharpe_col
                else 0.0
            )
            long_mdd = (
                pd.to_numeric(short_df[long_mdd_col], errors="coerce").mean()
                if long_mdd_col
                else 0.0
            )

            short_return_mean = float(short_return_series.mean())
            combined_return = long_return + short_return_mean

            short_num_col = next(
                (
                    c
                    for c in ["short_num_trades_current", "short_num_trades"]
                    if c in short_df.columns
                ),
                None,
            )
            short_num_mean = (
                pd.to_numeric(short_df[short_num_col], errors="coerce").mean()
                if short_num_col
                else 0.0
            )

            lines.append("## ロング単独 vs ロング+ショート 比較（全銘柄平均）")
            lines.append("")
            lines.append("| 指標 | total_return（ロング+ショート合算） | ロング+ショート |")
            lines.append("|------|-----------|----------------|")
            # total_returnにはショートPnL（cashベース）を含む合算値
            lines.append(f"| total_return | {long_return:.6f} | {combined_return:.6f} |")
            lines.append(f"| sharpe_ratio | {long_sharpe:.4f} | N/A（未計算）|")
            lines.append(f"| max_drawdown | {long_mdd:.6f} | N/A（未計算）|")
            lines.append(f"| short_return（平均） | 0.0 | {short_return_mean:.6f} |")
            lines.append(f"| short_num_trades（平均） | 0 | {short_num_mean:.1f} |")
            lines.append("")

    return "\n".join(lines)


def run_walk_forward_comparison_report(
    tasks: list,
    model_type: str = "XGBoostModel",
    source: str = "file",
    n_splits: int = 5,
    threshold: float = 0.0,
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
    slippage: float = 0.0,
    ensemble: bool = False,
    limit_symbols: Optional[int] = None,
) -> dict[str, Any]:
    """
    標準条件で Walk-Forward 検証を全銘柄実行し、比較レポートを保存する。

    tasks は呼び出し元（orchestration 等）で load_target_symbols() して渡す。

    Returns:
        保存先情報を含む辞書
    """
    targets = list(tasks)
    if limit_symbols is not None:
        targets = targets[:limit_symbols]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(
        ensure_dir(str(Path(get_results_dir()) / "backtest" / "walk_forward_reports"))
    )

    rows: list[dict[str, Any]] = []
    failed = 0

    for task in targets:
        market = task.market
        symbol = task.symbol
        logger.info(f"[WF] 実行開始: {market}/{symbol}")
        try:
            _, _, wf_df = run_backtest_walk_forward(
                market=market,
                symbol=symbol,
                model_type=model_type,
                task_name="return_regression",
                threshold=threshold,
                source=source,
                n_splits=n_splits,
                initial_cash=initial_cash,
                fee_rate=fee_rate,
                slippage=slippage,
                ensemble=ensemble,
            )
            if wf_df is None or wf_df.empty:
                failed += 1
                rows.append({"market": market, "symbol": symbol, "error": "wf_df is empty"})
                logger.warning(f"[WF] 空結果: {market}/{symbol}")
                continue

            summary = _summarize_wf_result(wf_df, market=market, symbol=symbol)
            rows.append(summary)
            # 実験記録（R-211）: WF実行ごとにパフォーマンスを記録する
            save_experiment_run(
                run_id=generate_run_id(),
                market=market,
                symbol=symbol,
                model_name=model_type,
                trained_at=ts,
                directional_accuracy=summary.get("win_rate"),
                params={
                    "n_splits": n_splits,
                    "threshold": threshold,
                    "initial_cash": initial_cash,
                    "fee_rate": fee_rate,
                    "slippage": slippage,
                    "ensemble": ensemble,
                    "total_return": summary.get("total_return"),
                    "sharpe_ratio": summary.get("sharpe_ratio"),
                },
            )
            logger.info(f"[WF] 実行完了: {market}/{symbol}")
        except Exception as exc:
            failed += 1
            rows.append({"market": market, "symbol": symbol, "error": str(exc)})
            logger.error(f"[WF] 実行失敗: {market}/{symbol}: {exc}", exc_info=True)

    current_df = pd.DataFrame(rows)
    summary_path = report_dir / f"wf_summary_{ts}.csv"
    current_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    previous_path = _find_latest_summary_csv(report_dir, exclude=summary_path)
    prev_df: Optional[pd.DataFrame] = None
    if previous_path is not None:
        try:
            prev_df = pd.read_csv(previous_path)
        except Exception as exc:
            logger.warning(f"前回サマリー読込失敗: {previous_path}: {exc}", exc_info=True)

    comparison_df = _build_comparison(current_df, prev_df)
    comparison_path = report_dir / f"wf_comparison_{ts}.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    markdown = _to_markdown_summary(comparison_df, previous_path)
    markdown_path = report_dir / f"wf_comparison_{ts}.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    err_col = current_df.get("error")
    success = len(current_df[err_col.isna()]) if err_col is not None else len(current_df)

    result = {
        "summary_path": str(summary_path),
        "comparison_path": str(comparison_path),
        "markdown_path": str(markdown_path),
        "previous_path": str(previous_path) if previous_path else None,
        "success": int(success),
        "failed": int(failed),
        "total": int(len(targets)),
    }

    logger.info(
        "Walk-Forward比較レポート完了: "
        f"success={result['success']} failed={result['failed']} total={result['total']}"
    )
    logger.info(f"保存先: {result['comparison_path']}")
    return result
