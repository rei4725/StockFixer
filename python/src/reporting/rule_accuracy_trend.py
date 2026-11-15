"""results/rule_accuracy の日ごとのルールの採点の JSON を読み返し、edge の流れの表を作る（#1837）。

書く側（scoring/rule_report.py）の JSON は1日1ファイルで、本体の読み手は最新の1ファイルの
全期間の窓しか読まない。ここでは読むだけで、日・市場・窓・ルール・向き・ホライズンごとに
n・hit_rate・baseline_rate・edge・mean_signed_return を1行にした表を作る。本番のジョブには
組み込まない。

- 読むのは accuracy_dir の直下の YYYY-MM-DD.json だけ（weekly/ の下は読まない）
- 値が null の組は NaN のまま残し、0 とは分ける（表では空で出す）
- 読めないファイル・形の違う JSON は警告を出して飛ばす。market は列で分ける
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

RULE_METRICS: tuple[str, ...] = (
    "hit_rate",
    "baseline_rate",
    "edge",
    "mean_signed_return",
)
_KEY_COLUMNS = ["date", "market", "window", "rule", "direction", "horizon"]
_COLUMNS = _KEY_COLUMNS + ["n"] + list(RULE_METRICS)
_DAY_FILE = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")


def load_rule_accuracy_reports(accuracy_dir: str) -> list[dict]:
    """accuracy_dir の直下の YYYY-MM-DD.json を日付の順に読む（読めないものは飛ばす）。"""
    reports: list[dict] = []
    for path in sorted(glob.glob(os.path.join(accuracy_dir, "*.json"))):
        name = os.path.basename(path)
        if not _DAY_FILE.match(name):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning("ルールの採点の JSON を読めず飛ばす: %s (%s)", path, e)
            continue
        if not isinstance(report, dict):
            logger.warning("ルールの採点の JSON の形が違うため飛ばす: %s", path)
            continue
        report["_file_date"] = os.path.splitext(name)[0]
        reports.append(report)
    return reports


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _rule_rows(base: dict[str, Any], rule: str, by_direction: Any) -> list[dict]:
    rows: list[dict] = []
    if not isinstance(by_direction, dict):
        return rows
    for direction, by_horizon in by_direction.items():
        if not isinstance(by_horizon, dict):
            continue
        for horizon, stats in by_horizon.items():
            if not isinstance(stats, dict):
                continue
            row = dict(base, rule=str(rule), direction=str(direction), horizon=str(horizon))
            row["n"] = _number(stats.get("n"))
            for metric in RULE_METRICS:
                row[metric] = _number(stats.get(metric))
            rows.append(row)
    return rows


def _report_rows(report: dict, market: str | None) -> list[dict]:
    day = str(report.get("_file_date", ""))
    report_market = report.get("market")
    if not isinstance(report_market, str):
        logger.warning("ルールの採点の JSON %s に market が無いため飛ばす", day)
        return []
    if market is not None and report_market != market:
        logger.warning(
            "ルールの採点の JSON %s の market が %s で %s と違うため飛ばす",
            day,
            report_market,
            market,
        )
        return []
    windows = report.get("windows")
    if not isinstance(windows, dict):
        logger.warning("ルールの採点の JSON %s に windows が無いため飛ばす", day)
        return []
    rows: list[dict] = []
    for window, section in windows.items():
        if not isinstance(section, dict) or not isinstance(section.get("rules"), dict):
            continue
        base = {"date": day, "market": report_market, "window": str(window)}
        for rule, by_direction in section["rules"].items():
            rows.extend(_rule_rows(base, rule, by_direction))
    return rows


def build_rule_accuracy_trend(reports: list[dict], market: str | None = None) -> pd.DataFrame:
    """日・市場・窓・ルール・向き・ホライズンごとの n と RULE_METRICS の表を返す。

    market を渡すと、その市場の JSON だけを使い、違う市場の JSON は警告を出して飛ばす。
    """
    rows = [row for report in reports for row in _report_rows(report, market)]
    df = pd.DataFrame(rows, columns=_COLUMNS)
    for column in ["n"] + list(RULE_METRICS):
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
    return df.sort_values(_KEY_COLUMNS, kind="stable").reset_index(drop=True)


def filter_rule_trend(
    df: pd.DataFrame,
    rule: str | None = None,
    window: str | None = None,
    horizon: str | None = None,
    direction: str | None = None,
) -> pd.DataFrame:
    """渡した条件の行だけに絞る（None の条件は絞らない）。"""
    out = df
    for column, value in (
        ("rule", rule),
        ("window", window),
        ("horizon", horizon),
        ("direction", direction),
    ):
        if value is not None:
            out = out[out[column] == str(value)]
    return out.reset_index(drop=True)


def load_rule_accuracy_trend(accuracy_dir: str, market: str | None = None) -> pd.DataFrame:
    """accuracy_dir の JSON を読み、表を返す。"""
    return build_rule_accuracy_trend(load_rule_accuracy_reports(accuracy_dir), market)


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.4f}"
    return str(value)


def format_rule_trend_markdown(df: pd.DataFrame) -> str:
    """表を Markdown の表の文字列にする（行が無ければ「データなし」）。null は空で出す。"""
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
