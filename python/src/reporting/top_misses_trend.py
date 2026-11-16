"""results/accuracy の採点の JSON の top_misses を日をまたいで集める（#1928）。

書く側（prediction/accuracy_report.py）は、モデル・ホライズンごとに外れの大きい銘柄を
top_misses に書くが、同じ銘柄が何日続けて外れたかを読み返す道が無かった。ここでは読むだけで、
モデル・ホライズン・market・symbol ごとに、出た日数・初めと最後の日・abs_error の平均と最大・
外れの向きの数を1行にした表を作る。本番のジョブには組み込まない。

- 形の違う要素（dict でない・symbol の無いもの）は飛ばす
- 値が null・数でないものは欠け（NaN）にし、平均と最大から外す
- 向き: 予想が実際より上（predicted_ratio > actual_ratio）を n_over、下を n_under と数える。
  どちらかが欠け・等しいときはどちらにも数えない
- 同じ日に同じ銘柄が2回出ても、出た日数は1日と数える（平均と最大と向きには両方を入れる）
"""

import math
from typing import Any

import pandas as pd

from src.reporting.accuracy_trend import load_accuracy_reports
from src.utils.logger import get_logger

logger = get_logger(__name__)

_GROUP_COLUMNS = ["model", "horizon", "market", "symbol"]
_MISS_COLUMNS = ["date"] + _GROUP_COLUMNS + ["predicted_ratio", "actual_ratio", "abs_error"]
TREND_COLUMNS: list[str] = _GROUP_COLUMNS + [
    "n_days",
    "first_date",
    "last_date",
    "mean_abs_error",
    "max_abs_error",
    "n_over",
    "n_under",
]


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    return float(value)


def _miss_rows(report: dict) -> list[dict]:
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
            misses = section.get("top_misses")
            if not isinstance(misses, list):
                continue
            for miss in misses:
                if not isinstance(miss, dict) or miss.get("symbol") in (None, ""):
                    continue
                rows.append(
                    {
                        "date": day,
                        "model": str(model),
                        "horizon": str(horizon),
                        "market": str(miss.get("market", "")),
                        "symbol": str(miss["symbol"]),
                        "predicted_ratio": _number(miss.get("predicted_ratio")),
                        "actual_ratio": _number(miss.get("actual_ratio")),
                        "abs_error": _number(miss.get("abs_error")),
                    }
                )
    return rows


def build_miss_rows(reports: list[dict]) -> pd.DataFrame:
    """日・モデル・ホライズン・銘柄ごとの top_misses の行の表を返す。"""
    rows = [row for report in reports for row in _miss_rows(report)]
    df = pd.DataFrame(rows, columns=_MISS_COLUMNS)
    for column in ["predicted_ratio", "actual_ratio", "abs_error"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
    return df


def build_top_misses_trend(reports: list[dict]) -> pd.DataFrame:
    """モデル・ホライズン・market・symbol ごとの1行の表を、出た日数の多い順に返す。"""
    misses = build_miss_rows(reports)
    if misses.empty:
        return pd.DataFrame(columns=TREND_COLUMNS)
    misses["over"] = (misses["predicted_ratio"] > misses["actual_ratio"]).astype(int)
    misses["under"] = (misses["predicted_ratio"] < misses["actual_ratio"]).astype(int)
    grouped = misses.groupby(_GROUP_COLUMNS, sort=True)
    out = grouped.agg(
        n_days=("date", "nunique"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        mean_abs_error=("abs_error", "mean"),
        max_abs_error=("abs_error", "max"),
        n_over=("over", "sum"),
        n_under=("under", "sum"),
    ).reset_index()
    for column in ["n_days", "n_over", "n_under"]:
        out[column] = out[column].astype(int)
    out = out.sort_values(
        ["n_days", "max_abs_error"] + _GROUP_COLUMNS,
        ascending=[False, False, True, True, True, True],
        kind="stable",
        na_position="last",
    )
    return out[TREND_COLUMNS].reset_index(drop=True)


def load_top_misses_trend(accuracy_dir: str) -> pd.DataFrame:
    """accuracy_dir の JSON を読み、top_misses の銘柄ごとの表を返す。"""
    return build_top_misses_trend(load_accuracy_reports(accuracy_dir))
