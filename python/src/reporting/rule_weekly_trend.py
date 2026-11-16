"""results/rule_accuracy/weekly の週ごとのルールの採点の JSON を読み返し、表を作る（#1906）。

書く側（scoring/rule_report.py の write_weekly_rule_accuracy）は、週ごとに重ならない採点と
その週の rule_best の出入りを weekly/YYYY-Www.json に書く。ここでは読むだけで、次の2つの表を
作る。本番のジョブには組み込まない。

- 採点の表: 週・市場・ルール・向き・ホライズンごとに n・hit_rate・baseline_rate・edge・
  mean_signed_return を1行
- 出入りの表: 週・市場ごとに n_outcomes と rule_best_diff の added・removed・changed の件数と
  evaluated_on・previous_evaluated_on を1行

- 読むのは weekly_dir の直下の YYYY-Www.json だけ
- 欠けた鍵・null は欠け（NaN・None）のまま残し、0 とは分ける（表では空で出す）
- 読めないファイル・形の違う JSON は警告を出して飛ばす。market は列で分ける
"""

import glob
import json
import math
import os
import re
from typing import Any

import pandas as pd

from src.reporting.rule_accuracy_trend import RULE_METRICS
from src.utils.logger import get_logger

logger = get_logger(__name__)

_OUTCOME_KEYS = ["week", "market", "rule", "direction", "horizon"]
OUTCOME_COLUMNS = _OUTCOME_KEYS + ["n"] + list(RULE_METRICS)
DIFF_COLUMNS = [
    "week",
    "market",
    "n_outcomes",
    "added",
    "removed",
    "changed",
    "evaluated_on",
    "previous_evaluated_on",
]
_DIFF_COUNTS = ("added", "removed", "changed")
_COUNT_COLUMNS = frozenset(("n", "n_outcomes") + _DIFF_COUNTS)
_WEEK_FILE = re.compile(r"^\d{4}-W\d{2}\.json$")


def load_weekly_reports(weekly_dir: str) -> list[dict]:
    """weekly_dir の直下の YYYY-Www.json を週の順に読む（読めないものは警告を出して飛ばす）。"""
    reports: list[dict] = []
    for path in sorted(glob.glob(os.path.join(weekly_dir, "*.json"))):
        name = os.path.basename(path)
        if not _WEEK_FILE.match(name):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning("週のルールの採点の JSON を読めず飛ばす: %s (%s)", path, e)
            continue
        if not isinstance(report, dict):
            logger.warning("週のルールの採点の JSON の形が違うため飛ばす: %s", path)
            continue
        report["_file_week"] = os.path.splitext(name)[0]
        reports.append(report)
    return reports


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _report_market(report: dict, market: str | None) -> str | None:
    """report の market を返す。無い・絞りの市場と違うときは警告を出して None。"""
    week = str(report.get("_file_week", ""))
    report_market = report.get("market")
    if not isinstance(report_market, str):
        logger.warning("週のルールの採点の JSON %s に market が無いため飛ばす", week)
        return None
    if market is not None and report_market != market:
        logger.warning(
            "週のルールの採点の JSON %s の market が %s で %s と違うため飛ばす",
            week,
            report_market,
            market,
        )
        return None
    return report_market


def _outcome_rows(report: dict, report_market: str) -> list[dict]:
    week = str(report.get("_file_week", ""))
    outcomes = report.get("outcomes")
    rows: list[dict] = []
    if not isinstance(outcomes, dict):
        return rows
    for rule, by_direction in outcomes.items():
        if not isinstance(by_direction, dict):
            continue
        for direction, by_horizon in by_direction.items():
            if not isinstance(by_horizon, dict):
                continue
            for horizon, stats in by_horizon.items():
                if not isinstance(stats, dict):
                    continue
                row: dict[str, Any] = {
                    "week": week,
                    "market": report_market,
                    "rule": str(rule),
                    "direction": str(direction),
                    "horizon": str(horizon),
                    "n": _number(stats.get("n")),
                }
                for metric in RULE_METRICS:
                    row[metric] = _number(stats.get(metric))
                rows.append(row)
    return rows


def _count(value: Any) -> float | None:
    return float(len(value)) if isinstance(value, list) else None


def _diff_row(report: dict, report_market: str) -> dict:
    diff = report.get("rule_best_diff")
    diff = diff if isinstance(diff, dict) else {}
    row: dict[str, Any] = {
        "week": str(report.get("_file_week", "")),
        "market": report_market,
        "n_outcomes": _number(report.get("n_outcomes")),
    }
    for key in _DIFF_COUNTS:
        row[key] = _count(diff.get(key))
    row["evaluated_on"] = _text(diff.get("evaluated_on"))
    row["previous_evaluated_on"] = _text(diff.get("previous_evaluated_on"))
    return row


def build_weekly_outcomes(reports: list[dict], market: str | None = None) -> pd.DataFrame:
    """週・市場・ルール・向き・ホライズンごとの n と RULE_METRICS の表を返す。"""
    rows: list[dict] = []
    for report in reports:
        report_market = _report_market(report, market)
        if report_market is not None:
            rows.extend(_outcome_rows(report, report_market))
    df = pd.DataFrame(rows, columns=OUTCOME_COLUMNS)
    for column in ["n"] + list(RULE_METRICS):
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
    return df.sort_values(_OUTCOME_KEYS, kind="stable").reset_index(drop=True)


def build_weekly_rule_best_diff(reports: list[dict], market: str | None = None) -> pd.DataFrame:
    """週・市場ごとの n_outcomes と rule_best の出入りの件数の表を返す。"""
    rows: list[dict] = []
    for report in reports:
        report_market = _report_market(report, market)
        if report_market is not None:
            rows.append(_diff_row(report, report_market))
    df = pd.DataFrame(rows, columns=DIFF_COLUMNS)
    for column in ("n_outcomes",) + _DIFF_COUNTS:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
    return df.sort_values(["week", "market"], kind="stable").reset_index(drop=True)


def filter_weekly_outcomes(
    df: pd.DataFrame,
    rule: str | None = None,
    horizon: str | None = None,
    direction: str | None = None,
) -> pd.DataFrame:
    """渡した条件の行だけに絞る（None の条件は絞らない）。"""
    out = df
    for column, value in (("rule", rule), ("horizon", horizon), ("direction", direction)):
        if value is not None:
            out = out[out[column] == str(value)]
    return out.reset_index(drop=True)


def load_weekly_trend(
    weekly_dir: str, market: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """weekly_dir の JSON を読み、(採点の表, 出入りの表) を返す。"""
    reports = load_weekly_reports(weekly_dir)
    return build_weekly_outcomes(reports, market), build_weekly_rule_best_diff(reports, market)


def _cell(column: str, value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.0f}" if column in _COUNT_COLUMNS else f"{value:.4f}"
    return str(value)


def format_weekly_markdown(df: pd.DataFrame) -> str:
    """表を Markdown の表の文字列にする（行が無ければ「データなし」）。欠けは空で出す。"""
    if df.empty:
        return "データなし"
    columns = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in df.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(c, v) for c, v in zip(columns, values)) + " |")
    return "\n".join(lines)
