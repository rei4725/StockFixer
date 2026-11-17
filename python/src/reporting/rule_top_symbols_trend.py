"""results/rule_accuracy のルールの採点の JSON の top_symbols を日をまたいで集める（#1972）。

書く側（scoring/rule_report.py の _top_symbols）は、h5 の採点を銘柄・ルール・向きでまとめ、
件数の多い順に top_symbols を書くが、同じ組が日をまたいでどう動いたかを読み返す道が無かった。
ここでは読むだけで、market・symbol・rule_name・direction ごとに、出た日数・初めと最後の日・
最後の日の n_h5・hit_rate_h5・edge_h5・edge_h5 の最小と最大を1行にした表を作る。
本番のジョブには組み込まない。

- 日はファイル名の日付（load_rule_accuracy_reports の _file_date）を使う。market は JSON の上の段
- 形の違う要素（dict でない・symbol の無いもの）は飛ばす
- 値が null・数でないもの（bool も）は欠け（NaN）にし、最小と最大から外す
- 最後の日の値は、最後の日の要素の値をそのまま使う（欠けなら NaN のまま）。同じ日に同じ組が
  2回出たときは、後に書かれた要素を使う
"""

import math
from typing import Any

import pandas as pd

from src.reporting.rule_accuracy_trend import load_rule_accuracy_reports
from src.utils.logger import get_logger

logger = get_logger(__name__)

_GROUP_COLUMNS = ["market", "symbol", "rule_name", "direction"]
_VALUE_COLUMNS = ["n_h5", "hit_rate_h5", "edge_h5"]
_ROW_COLUMNS = ["date"] + _GROUP_COLUMNS + _VALUE_COLUMNS
TREND_COLUMNS: list[str] = _GROUP_COLUMNS + [
    "n_days",
    "first_date",
    "last_date",
    "last_n_h5",
    "last_hit_rate_h5",
    "last_edge_h5",
    "min_edge_h5",
    "max_edge_h5",
]


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    return float(value)


def _top_symbol_rows(report: dict) -> list[dict]:
    day = str(report.get("_file_date", report.get("date", "")))
    items = report.get("top_symbols")
    if not isinstance(items, list):
        logger.warning("ルールの採点の JSON %s に top_symbols が無いため飛ばす", day)
        return []
    market = str(report.get("market", ""))
    rows: list[dict] = []
    for item in items:
        if not isinstance(item, dict) or item.get("symbol") in (None, ""):
            continue
        row: dict[str, Any] = {
            "date": day,
            "market": market,
            "symbol": str(item["symbol"]),
            "rule_name": str(item.get("rule_name", "")),
            "direction": str(item.get("direction", "")),
        }
        for column in _VALUE_COLUMNS:
            row[column] = _number(item.get(column))
        rows.append(row)
    return rows


def build_top_symbol_rows(reports: list[dict]) -> pd.DataFrame:
    """日・market・symbol・rule_name・direction ごとの top_symbols の行の表を返す。"""
    rows = [row for report in reports for row in _top_symbol_rows(report)]
    df = pd.DataFrame(rows, columns=_ROW_COLUMNS)
    for column in _VALUE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
    return df


def build_rule_top_symbols_trend(reports: list[dict]) -> pd.DataFrame:
    """market・symbol・rule_name・direction ごとの1行の表を、出た日数の多い順に返す。"""
    rows = build_top_symbol_rows(reports)
    if rows.empty:
        return pd.DataFrame(columns=TREND_COLUMNS)
    rows = rows.sort_values("date", kind="stable")
    grouped = rows.groupby(_GROUP_COLUMNS, sort=True)
    out = grouped.agg(
        n_days=("date", "nunique"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        min_edge_h5=("edge_h5", "min"),
        max_edge_h5=("edge_h5", "max"),
    ).reset_index()
    last = rows.drop_duplicates(_GROUP_COLUMNS, keep="last")[_GROUP_COLUMNS + _VALUE_COLUMNS]
    last = last.rename(columns={column: f"last_{column}" for column in _VALUE_COLUMNS})
    out = out.merge(last, on=_GROUP_COLUMNS, how="left")
    out["n_days"] = out["n_days"].astype(int)
    out = out.sort_values(
        ["n_days", "last_n_h5"] + _GROUP_COLUMNS,
        ascending=[False, False, True, True, True, True],
        kind="stable",
        na_position="last",
    )
    return out[TREND_COLUMNS].reset_index(drop=True)


def load_rule_top_symbols_trend(accuracy_dir: str) -> pd.DataFrame:
    """accuracy_dir のルールの採点の JSON を読み、top_symbols の組ごとの表を返す。"""
    return build_rule_top_symbols_trend(load_rule_accuracy_reports(accuracy_dir))
