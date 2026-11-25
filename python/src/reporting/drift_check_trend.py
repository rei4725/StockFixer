"""results/drift_check の日次ドリフトチェックの記録を読み返し、日ごとに1行の表を作る（#2050）。

書く側（orchestration/jobs/drift_record.py）は1日1件の JSON で、閾値超過の割合や
再学習の数の流れを見るにはファイルを1つずつ開くしかなかった。ここでは読むだけで、
日ごとに1行の表（DataFrame）にする。本番のジョブには組み込まない。

- 名前が YYYY-MM-DD.json でないファイル、読めないファイル、壊れた JSON、
  オブジェクトでない JSON は警告を出して飛ばす
- 無い項目は空（None）にする
"""

import glob
import json
import math
import os
import re
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

TREND_COLUMNS: list[str] = [
    "date",
    "n_evaluated",
    "n_triggered",
    "triggered_share",
    "warn_share",
    "warn",
    "mae_threshold",
    "hit_rate_threshold",
    "fresh_skipped",
    "retrain_targets",
    "retrain_success",
    "n_triggered_jp",
    "n_triggered_us",
    "n_mae_over",
    "n_hit_rate_over",
    "n_repeated",
    "retrain_seconds",
]

# 超えた銘柄の一覧（triggered_symbols）から数える列（#2167）。一覧の無い記録は空にし 0 と分ける
_DETAIL_COUNT_COLUMNS: list[str] = [
    "n_triggered_jp",
    "n_triggered_us",
    "n_mae_over",
    "n_hit_rate_over",
    "n_repeated",
]

_DATE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$")


def _triggered_entries(data: dict) -> list[dict] | None:
    """記録の超えた銘柄の一覧（無い・形が違うなら None）。"""
    entries = data.get("triggered_symbols")
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        return None
    return entries


def _symbol_keys(entries: list[dict]) -> set[tuple[str, str]]:
    return {(str(e.get("market")), str(e.get("symbol"))) for e in entries}


def _detail_counts(entries: list[dict] | None, previous: list[dict] | None) -> dict:
    """一覧から数える列。previous は前の記録の一覧で、無ければくり返しの数を空にする。"""
    counts: dict[str, int | None] = dict.fromkeys(_DETAIL_COUNT_COLUMNS)
    if entries is None:
        return counts
    counts["n_triggered_jp"] = sum(e.get("market") == "jp" for e in entries)
    counts["n_triggered_us"] = sum(e.get("market") == "us" for e in entries)
    counts["n_mae_over"] = sum(e.get("mae_over") is True for e in entries)
    counts["n_hit_rate_over"] = sum(e.get("hit_rate_over") is True for e in entries)
    if previous is not None:
        counts["n_repeated"] = len(_symbol_keys(entries) & _symbol_keys(previous))
    return counts


def load_drift_check_records(record_dir: str) -> list[dict]:
    """record_dir の YYYY-MM-DD.json を日の順に読む（読めないものは飛ばす）。"""
    rows: list[dict] = []
    previous: list[dict] | None = None
    for path in sorted(glob.glob(os.path.join(record_dir, "*.json"))):
        match = _DATE_PATTERN.match(os.path.basename(path))
        if match is None:
            logger.warning("ドリフトチェックの記録の名前の形が違うため飛ばす: %s", path)
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.warning("ドリフトチェックの記録を読めず飛ばす: %s (%s)", path, e)
            continue
        if not isinstance(data, dict):
            logger.warning("ドリフトチェックの記録の形が違うため飛ばす: %s", path)
            continue
        row = {column: data.get(column) for column in TREND_COLUMNS}
        row["date"] = match.group(1)
        entries = _triggered_entries(data)
        row.update(_detail_counts(entries, previous))
        previous = entries
        rows.append(row)
    return rows


def build_drift_check_trend(rows: list[dict]) -> pd.DataFrame:
    """日ごとの行を日の順の表にする。"""
    df = pd.DataFrame(rows, columns=TREND_COLUMNS)
    # 空が混じっても 16.0 でなく 16 と出るよう、欠けを持てる整数の列にする
    df = df.astype(dict.fromkeys(_DETAIL_COUNT_COLUMNS, "Int64"))
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def load_drift_check_trend(record_dir: str) -> pd.DataFrame:
    """record_dir の記録を読み、日ごとの表を返す。"""
    return build_drift_check_trend(load_drift_check_records(record_dir))


def _cell(value: Any) -> str:
    if value is None or value is pd.NA:
        return "null"
    if isinstance(value, float):
        # 欠けのある数の列は float になり、欠けは NaN で来る
        return "null" if math.isnan(value) else f"{value:.4g}"
    return str(value)


def format_drift_check_markdown(df: pd.DataFrame) -> str:
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


def format_drift_check_csv(df: pd.DataFrame) -> str:
    """表を CSV の文字列にする（欠けは null と書き、False・0 と分ける）。"""
    return df.to_csv(index=False, na_rep="null")
