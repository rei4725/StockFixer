"""results/promotion の週の昇格ゲートの判定を読み返し、流れの表を作る（#2045）。

書く側（prediction/promotion_gate.py の save_promotion_result）の JSON は1回1ファイルで、
どの基準が何週続けて challenger を止めているかを見るにはファイルを1つずつ開くしかなかった。
ここでは読むだけで、回ごとに1行の表を作る。本番のジョブには組み込まない。

- 読むのは YYYYMMDD_HHMMSS_promotion_result.json だけで、名前の日時の順に並べる
- 読めないファイル・形の違う JSON は警告を出して飛ばす
- passed が null（WF を比べられない回など）と欄の無いものは欠け（pd.NA）にし、False と分ける
- mdd の値は shadow_abs・current_abs を shadow・current の列に入れる
- 閾値は threshold を、無ければ floor を <基準>_threshold の列に入れる
- <基準>_fail_streak は、その回までに passed が False の続いた回数である。True で 0 に戻る。
  passed が欠けの回は、その回の列を欠けにし、数は戻さずに次の回へ持ち越す
"""

import json
import os
import re
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_FILE_PATTERN = re.compile(r"^(\d{8}_\d{6})_promotion_result\.json$")

CRITERIA: tuple[str, ...] = (
    "net_return",
    "mdd",
    "sharpe",
    "hit_rate",
    "slippage",
    "prediction_spread",
)
RUN_COLUMNS: tuple[str, ...] = ("evaluated_at", "shadow_model", "current_model", "eligible")
VALUE_FIELDS: tuple[str, ...] = ("shadow", "current", "threshold")
PASSED_COLUMNS: tuple[str, ...] = tuple(f"{name}_passed" for name in CRITERIA)
VALUE_COLUMNS: tuple[str, ...] = tuple(
    f"{name}_{field}" for name in CRITERIA for field in VALUE_FIELDS
)
STREAK_COLUMNS: tuple[str, ...] = tuple(f"{name}_fail_streak" for name in CRITERIA)
# 値の欄の別名（mdd は絶対値で書かれ、hit_rate の閾値は floor で書かれる）
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "shadow": ("shadow", "shadow_abs"),
    "current": ("current", "current_abs"),
    "threshold": ("threshold", "floor"),
}


def load_promotion_gate_records(results_dir: str) -> list[dict]:
    """results_dir の回ごとの JSON を、ファイル名の日時の順に読む（読めないものは飛ばす）。"""
    try:
        names = os.listdir(results_dir)
    except OSError as e:
        logger.warning("昇格ゲートの記録の置き場を読めない: %s (%s)", results_dir, e)
        return []
    records: list[dict] = []
    for name in sorted(n for n in names if _FILE_PATTERN.match(n)):
        path = os.path.join(results_dir, name)
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning("昇格ゲートの記録の JSON を読めず飛ばす: %s (%s)", path, e)
            continue
        if not isinstance(record, dict):
            logger.warning("昇格ゲートの記録の JSON の形が違うため飛ばす: %s", path)
            continue
        criteria = record.get("criteria")
        if criteria is not None and not isinstance(criteria, dict):
            logger.warning("昇格ゲートの記録の criteria の形が違うため飛ばす: %s", path)
            continue
        record["_file"] = os.path.splitext(name)[0]
        records.append(record)
    return records


def _number(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return pd.NA
    return value


def _flag(value: Any) -> Any:
    return value if isinstance(value, bool) else pd.NA


def _text(value: Any) -> Any:
    return value if isinstance(value, str) else pd.NA


def _field(criterion: dict, field: str) -> Any:
    for key in _FIELD_ALIASES[field]:
        if key in criterion:
            return _number(criterion.get(key))
    return pd.NA


def _row(record: dict) -> dict[str, Any]:
    row: dict[str, Any] = {"file": record["_file"]}
    row["evaluated_at"] = _text(record.get("evaluated_at"))
    row["shadow_model"] = _text(record.get("shadow_model"))
    row["current_model"] = _text(record.get("current_model"))
    row["eligible"] = _flag(record.get("eligible"))
    criteria = record.get("criteria")
    criteria = criteria if isinstance(criteria, dict) else {}
    for name in CRITERIA:
        criterion = criteria.get(name)
        criterion = criterion if isinstance(criterion, dict) else {}
        row[f"{name}_passed"] = _flag(criterion.get("passed"))
        for field in VALUE_FIELDS:
            row[f"{name}_{field}"] = _field(criterion, field)
    row["wf_source_file"] = _text(record.get("wf_source_file"))
    return row


def _add_fail_streaks(rows: list[dict[str, Any]]) -> None:
    for name in CRITERIA:
        streak = 0
        for row in rows:
            passed = row[f"{name}_passed"]
            if passed is pd.NA:
                row[f"{name}_fail_streak"] = pd.NA
                continue
            streak = 0 if passed else streak + 1
            row[f"{name}_fail_streak"] = streak


def build_promotion_gate_trend(records: list[dict]) -> pd.DataFrame:
    """回ごとに1行の表を返す（基準ごとの合否・値・閾値と、続けて落ちた回数）。"""
    rows = [_row(record) for record in records]
    _add_fail_streaks(rows)
    columns: list[str] = ["file"] + list(RUN_COLUMNS)
    for name in CRITERIA:
        columns += [f"{name}_passed"] + [f"{name}_{field}" for field in VALUE_FIELDS]
    columns += ["wf_source_file"] + list(STREAK_COLUMNS)
    df = pd.DataFrame(rows, columns=columns)
    for column in ("eligible",) + PASSED_COLUMNS:
        df[column] = df[column].astype("boolean")
    for column in VALUE_COLUMNS:
        df[column] = df[column].astype("Float64")
    for column in STREAK_COLUMNS:
        df[column] = df[column].astype("Int64")
    return df.reset_index(drop=True)


def load_promotion_gate_trend(results_dir: str) -> pd.DataFrame:
    """results_dir の JSON を読み、回ごとの表を返す。"""
    return build_promotion_gate_trend(load_promotion_gate_records(results_dir))


def _cell(value: Any) -> str:
    if value is None or value is pd.NA:
        return "null"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def format_promotion_gate_markdown(df: pd.DataFrame) -> str:
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


def format_promotion_gate_csv(df: pd.DataFrame) -> str:
    """表を CSV の文字列にする（欠けは null と書き、False・0 と分ける）。"""
    return df.to_csv(index=False, na_rep="null")
