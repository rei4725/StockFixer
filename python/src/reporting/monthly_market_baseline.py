"""月報の「市場別の的中率と基準」の節（#1816）。

全体の Hit Rate 1つでは、片方の市場が多数派の基準（常に多い方の向きを言った場合の的中率）
より下でも隠れる。読み込み済みの窓の採点を市場ごとに分け、基準の物差し（#1309 の
baseline_metrics）で件数・的中率・多数派の基準・差・rank_corr を出す。

reporting は prediction を import できない（BC の独立）。物差しの関数は合成ルートが渡す。
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from src.reporting.kpi import _REPORT_DAYS, predicted_jst_date, window_start_jst

BaselineFn = Callable[[pd.DataFrame], dict]


def _window_rows(accuracy_df: pd.DataFrame, days: int) -> pd.DataFrame:
    """月報の Hit Rate と同じ行（production・predicted_at の JST の日付が窓の中）に絞る。"""
    df = accuracy_df
    if "model_name" in df.columns:
        df = df[df["model_name"] == "production"]
    if "predicted_at" in df.columns:
        df = df[predicted_jst_date(df["predicted_at"]) >= window_start_jst(days)]
    return df


def _pct(val: Optional[float]) -> str:
    return f"{val * 100:.2f}%" if val is not None else "N/A"


def _pt(val: Optional[float]) -> str:
    return f"{val * 100:+.2f}pt" if val is not None else "N/A"


def _f2(val: Optional[float]) -> str:
    return f"{val:.2f}" if val is not None else "N/A"


def _market_row(market: str, metrics: dict) -> str:
    majority = metrics.get("majority_baseline_accuracy")
    diff = metrics.get("hit_rate_minus_majority")
    hit = majority + diff if majority is not None and diff is not None else None
    return (
        f"| {market} | {metrics.get('n_baseline', 0)} | {_pct(hit)} | {_pct(majority)} "
        f"| {_pt(diff)} | {_f2(metrics.get('rank_corr'))} |"
    )


def market_baseline_lines(
    accuracy_df: Optional[pd.DataFrame],
    baseline_fn: Optional[BaselineFn],
    days: int = _REPORT_DAYS,
) -> list[str]:
    """節の Markdown の行を返す。

    的中率は baseline_fn と同じ行（予想と実際の変化率がそろった行）での向きの的中率で、
    多数派の基準との差は baseline_fn の hit_rate_minus_majority をそのまま出す。
    """
    lines = [f"## 市場別の的中率と基準（直近{days}日）", ""]
    if baseline_fn is None:
        return lines + ["基準の物差しが渡されていないため出さない", ""]
    if accuracy_df is None or accuracy_df.empty or "market" not in accuracy_df.columns:
        return lines + ["採点データなし", ""]
    df = _window_rows(accuracy_df, days)
    rows = []
    for market, group in df.groupby("market", sort=True):
        metrics = baseline_fn(group)
        if metrics.get("n_baseline", 0) > 0:
            rows.append(_market_row(str(market), metrics))
    if not rows:
        return lines + ["採点データなし", ""]
    return lines + [
        "| 市場 | 件数 | 的中率 | 多数派の基準 | 差 | rank_corr |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
        "多数派の基準は、窓の中で常に多い方の向きを言った場合の的中率。差が負の市場は基準に負けている。",
        "",
    ]
