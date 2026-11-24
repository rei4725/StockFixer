"""results/unified_training の統合モデルの学習の記録を読み返し、1回の学習ごとに1行の表を作る（#2151）。

書く側（prediction/training_record.py）は1回の学習ごとに1件の JSON で、予想の上がる割合と
実際の割合のずれを見るにはファイルを1つずつ開いて引き算するしかなかった。ここでは読むだけで、
1回の学習ごとに1行の表（DataFrame）にする。本番のジョブには組み込まない。

- 名前が YYYY-MM-DD_<model_name>.json でないファイル、読めないファイル、壊れた JSON、
  オブジェクトでない JSON は警告を出して飛ばす
- 無い項目は空（None）にする。ずれはどちらかの割合が無ければ None にする
"""

import glob
import json
import math
import os
import re
from typing import Any, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

TREND_COLUMNS: list[str] = [
    "date",
    "model_name",
    "recorded_at",
    "best_iteration",
    "trees",
    "val_rows",
    "val_pred_std",
    "is_constant",
    "val_predicted_up_rate",
    "val_actual_up_rate",
    "up_rate_gap",
]

_NAME_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})_(.+)\.json$")


def _up_rate_gap(predicted: Any, actual: Any) -> Optional[float]:
    """予想の割合 − 実際の割合。どちらかが数でなければ None。"""
    for value in (predicted, actual):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
    return float(predicted - actual)


def load_unified_training_records(record_dir: str) -> list[dict]:
    """record_dir の YYYY-MM-DD_<model_name>.json を読む（読めないものは飛ばす）。"""
    rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(record_dir, "*.json"))):
        match = _NAME_PATTERN.match(os.path.basename(path))
        if match is None:
            logger.warning("統合モデルの学習の記録の名前の形が違うため飛ばす: %s", path)
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.warning("統合モデルの学習の記録を読めず飛ばす: %s (%s)", path, e)
            continue
        if not isinstance(data, dict):
            logger.warning("統合モデルの学習の記録の形が違うため飛ばす: %s", path)
            continue
        row = {column: data.get(column) for column in TREND_COLUMNS}
        row["date"] = match.group(1)
        row["model_name"] = match.group(2)
        row["up_rate_gap"] = _up_rate_gap(row["val_predicted_up_rate"], row["val_actual_up_rate"])
        rows.append(row)
    return rows


def build_unified_training_trend(rows: list[dict]) -> pd.DataFrame:
    """1回の学習ごとの行を、日付・モデル名の順の表にする。

    整数の列が欠けで float に変わらないよう、列は object のまま持つ。
    """
    df = pd.DataFrame(rows, columns=TREND_COLUMNS, dtype=object)
    return df.sort_values(["date", "model_name"], kind="stable").reset_index(drop=True)


def load_unified_training_trend(record_dir: str, model_name: Optional[str] = None) -> pd.DataFrame:
    """record_dir の記録を読み、表を返す。model_name を渡すとそのモデルだけにする。"""
    df = build_unified_training_trend(load_unified_training_records(record_dir))
    if model_name is not None:
        df = df[df["model_name"] == model_name].reset_index(drop=True)
    return df


def _cell(value: Any) -> str:
    if value is None or value is pd.NA:
        return "null"
    if isinstance(value, float):
        return "null" if math.isnan(value) else f"{value:.4g}"
    return str(value)


def format_unified_training_markdown(df: pd.DataFrame) -> str:
    """表を Markdown の表の文字列にする（欠けは null。行が無ければ「データなし」）。"""
    if df.empty:
        return "データなし"
    columns = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in df.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(v) for v in values) + " |")
    return "\n".join(lines)


def format_unified_training_csv(df: pd.DataFrame) -> str:
    """表を CSV の文字列にする（欠けは null と書き、False・0 と分ける）。"""
    return df.to_csv(index=False, na_rep="null")
