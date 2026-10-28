"""向きの的中率と並べて見る物差し（#1309）。

direction_match は向きだけで決まるため、予想がほとんど「上がる」のときは、的中率が
その期間に実際に上がった割合をほぼそのまま測る。腕と相場の向きを見分けるため、
次の物差しを出す（direction_match の意味は変えない）。

- actual_up_rate: 実際に上がった割合（actual_ratio >= 0。direction_match と同じ境）
- predicted_up_rate: 上がると予想した割合（predicted_ratio >= 0）
- majority_baseline_accuracy: 常に多い方を言った場合の的中率
- hit_rate_minus_up_rate: 的中率 - 実際に上がった割合（全部「上がる」と言うと 0）
- hit_rate_minus_majority: 的中率 - 常に多い方を言った場合の的中率
- mae: |predicted_ratio - actual_ratio| の平均
- mean_abs_predicted_ratio / mean_abs_actual_ratio: 予想と実際の大きさの平均
- rank_corr: predicted_ratio と actual_ratio の順位相関（Spearman）
- abs_rank_corr: |predicted_ratio| と |actual_ratio| の順位相関（Spearman）
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

BASELINE_METRIC_KEYS: tuple[str, ...] = (
    "n_baseline",
    "actual_up_rate",
    "predicted_up_rate",
    "majority_baseline_accuracy",
    "hit_rate_minus_up_rate",
    "hit_rate_minus_majority",
    "mae",
    "mean_abs_predicted_ratio",
    "mean_abs_actual_ratio",
    "rank_corr",
    "abs_rank_corr",
)


def _empty_metrics() -> dict:
    metrics: dict = {key: None for key in BASELINE_METRIC_KEYS}
    metrics["n_baseline"] = 0
    return metrics


def _spearman(x: pd.Series, y: pd.Series) -> Optional[float]:
    """順位の Pearson 相関を返す（scipy を使わない）。値が1種類しかなければ None。"""
    if len(x) < 2:
        return None
    rx = x.rank(method="average")
    ry = y.rank(method="average")
    if rx.nunique() < 2 or ry.nunique() < 2:
        return None
    val = rx.corr(ry)
    return float(val) if not pd.isna(val) else None


def baseline_metrics(rows: pd.DataFrame) -> dict:
    """予想の変化率と実際の変化率がそろった行から、基準の的中率などを返す。

    Args:
        rows: predicted_ratio と actual_ratio の列を持つ prediction_accuracy の行

    Returns:
        BASELINE_METRIC_KEYS の dict（行が無いときは n_baseline=0、他は None）
    """
    if rows is None or rows.empty:
        return _empty_metrics()
    if "predicted_ratio" not in rows.columns or "actual_ratio" not in rows.columns:
        return _empty_metrics()
    pred = pd.to_numeric(rows["predicted_ratio"], errors="coerce")
    actual = pd.to_numeric(rows["actual_ratio"], errors="coerce")
    valid = pred.notna() & actual.notna()
    pred = pred[valid]
    actual = actual[valid]
    if pred.empty:
        return _empty_metrics()

    predicted_up = pred >= 0
    actual_up = actual >= 0
    hit_rate = float((predicted_up == actual_up).mean())
    up_rate = float(actual_up.mean())
    majority = max(up_rate, 1.0 - up_rate)
    return {
        "n_baseline": int(len(pred)),
        "actual_up_rate": up_rate,
        "predicted_up_rate": float(predicted_up.mean()),
        "majority_baseline_accuracy": majority,
        "hit_rate_minus_up_rate": hit_rate - up_rate,
        "hit_rate_minus_majority": hit_rate - majority,
        "mae": float((pred - actual).abs().mean()),
        "mean_abs_predicted_ratio": float(pred.abs().mean()),
        "mean_abs_actual_ratio": float(actual.abs().mean()),
        "rank_corr": _spearman(pred, actual),
        "abs_rank_corr": _spearman(pred.abs(), actual.abs()),
    }
