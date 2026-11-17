"""採点の JSON の業種・予想の大きさの内訳を日をまたいで並べる（#1927）。

書く側（prediction/accuracy_report.py）はモデル・ホライズンごとに by_sector（業種名ごと）と
by_magnitude（|予想| の五分位 q1〜q5）を毎日書くが、読み返す道が無かった。ここでは読むだけで、
日・モデル・ホライズン・内訳の種類・区分ごとの表と、区分ごとの直近 N 日の n_scored で
重みを付けた平均を作る。本番のジョブには組み込まない。

- ファイルの読み込みは accuracy_trend.load_accuracy_reports をそのまま使う
- 内訳の無い日・空の内訳（{}）は行を作らない。値が null・鍵の無い値は欠け（NaN）にする
- 重み付きの平均の N 日は、モデル・ホライズンの組ごとに、その内訳の行がある日の直近 N 日。
  物差しの値と n_scored の両方が読め、n_scored が正の行だけを平均に入れる
"""

import math
from typing import Any

import pandas as pd

from src.reporting.accuracy_trend import load_accuracy_reports
from src.utils.logger import get_logger

logger = get_logger(__name__)

BREAKDOWN_KINDS: dict[str, str] = {"sector": "by_sector", "magnitude": "by_magnitude"}
BREAKDOWN_METRICS: tuple[str, ...] = (
    "direction_accuracy",
    "majority_baseline_accuracy",
    "hit_rate_minus_majority",
    "predicted_up_rate",
    "rank_corr",
)
_GROUP_COLUMNS = ["model", "horizon", "kind", "bucket"]
_KEY_COLUMNS = ["date"] + _GROUP_COLUMNS
_BASE_COLUMNS = _KEY_COLUMNS + ["n_scored"] + list(BREAKDOWN_METRICS)
_SUMMARY_COLUMNS = (
    _GROUP_COLUMNS
    + ["first_date", "last_date", "n_days", "n_scored_total"]
    + [f"{metric}_wavg" for metric in BREAKDOWN_METRICS]
)


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    return float(value)


def _report_rows(report: dict, kind: str) -> list[dict]:
    key = BREAKDOWN_KINDS[kind]
    day = str(report.get("date", ""))
    models = report.get("models")
    if not isinstance(models, dict):
        logger.warning("採点の JSON %s に models が無いため飛ばす", day)
        return []
    rows: list[dict] = []
    for model, by_horizon in models.items():
        if not isinstance(by_horizon, dict):
            continue
        for horizon, section in by_horizon.items():
            if not isinstance(section, dict):
                continue
            breakdown = section.get(key)
            if not isinstance(breakdown, dict):
                continue
            for bucket, bucket_section in sorted(breakdown.items()):
                if not isinstance(bucket_section, dict):
                    continue
                row: dict[str, Any] = {
                    "date": day,
                    "model": str(model),
                    "horizon": str(horizon),
                    "kind": kind,
                    "bucket": str(bucket),
                    "n_scored": _number(bucket_section.get("n_scored")),
                }
                for metric in BREAKDOWN_METRICS:
                    row[metric] = _number(bucket_section.get(metric))
                rows.append(row)
    return rows


def build_breakdown_trend(reports: list[dict], kind: str) -> pd.DataFrame:
    """日・モデル・ホライズン・内訳の種類・区分ごとの n_scored と物差しの表を返す。"""
    if kind not in BREAKDOWN_KINDS:
        raise ValueError(f"kind は {sorted(BREAKDOWN_KINDS)} のどれか: {kind}")
    rows = [row for report in reports for row in _report_rows(report, kind)]
    df = pd.DataFrame(rows, columns=_BASE_COLUMNS)
    for column in ["n_scored"] + list(BREAKDOWN_METRICS):
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
    return df.sort_values(_KEY_COLUMNS, kind="stable").reset_index(drop=True)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    usable = values.notna() & weights.notna() & (weights > 0)
    total = float(weights[usable].sum())
    if total <= 0:
        return math.nan
    return float((values[usable] * weights[usable]).sum() / total)


def summarize_breakdown_trend(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """区分ごとの直近 window 日の n_scored で重みを付けた平均の表を返す。

    直近 window 日は、モデル・ホライズンの組ごとに行のある日を新しい順に数える。
    列の <物差し>_wavg は、値と n_scored の読める行の sum(値×n_scored)/sum(n_scored)。
    """
    if window < 1:
        raise ValueError(f"window は1以上: {window}")
    summaries: list[dict] = []
    for _, pair in df.groupby(["model", "horizon"], sort=True):
        recent_days = sorted(pair["date"].unique())[-window:]
        recent = pair[pair["date"].isin(recent_days)]
        for keys, group in recent.groupby(_GROUP_COLUMNS, sort=True):
            summary: dict[str, Any] = dict(zip(_GROUP_COLUMNS, keys))
            summary.update(
                {
                    "first_date": str(group["date"].min()),
                    "last_date": str(group["date"].max()),
                    "n_days": int(group["date"].nunique()),
                    "n_scored_total": float(group["n_scored"].sum(min_count=1)),
                }
            )
            for metric in BREAKDOWN_METRICS:
                summary[f"{metric}_wavg"] = _weighted_mean(group[metric], group["n_scored"])
            summaries.append(summary)
    return pd.DataFrame(summaries, columns=_SUMMARY_COLUMNS)


def load_breakdown_trend(accuracy_dir: str, kind: str) -> pd.DataFrame:
    """accuracy_dir の JSON を読み、内訳の日ごとの表を返す。"""
    return build_breakdown_trend(load_accuracy_reports(accuracy_dir), kind)
