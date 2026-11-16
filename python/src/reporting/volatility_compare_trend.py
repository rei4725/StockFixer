"""results/backtest_compare/volatility_sizing の週ごとの比べの JSON を読み返し、表を作る（#1864）。

書く側（backtest/volatility_compare.py）の JSON は1週1ファイルで、fixed と各形の差を読むには
週ごと・市場ごとにファイルを開いて引き算するしかなかった。ここでは読むだけで、週・市場・形・k
ごとの1行の表に、同じ週・同じ市場の fixed との差を足す。本番のジョブには組み込まない。

- 差は total_return・max_drawdown・return_over_volatility の3つで、<値> - <fixed の値>
- fixed の無い週・市場、または値の無い（None の）ときは差を空（NaN）にする
- 読めないファイル・形の違う JSON は警告を出して飛ばす
"""

import glob
import json
import math
import os
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

FORM_METRICS: tuple[str, ...] = (
    "total_return",
    "max_drawdown",
    "return_over_volatility",
    "num_trades",
    "stop_outs",
)
DIFF_METRICS: tuple[str, ...] = ("total_return", "max_drawdown", "return_over_volatility")
FIXED_FORM = "fixed"
_KEY_COLUMNS = ["week", "market", "form", "k"]
_WEEK_COLUMNS = ["period_start", "period_end", "symbols", "symbols_failed"]
_DIFF_COLUMNS = [f"{metric}_vs_fixed" for metric in DIFF_METRICS]
TREND_COLUMNS = _KEY_COLUMNS + _WEEK_COLUMNS + list(FORM_METRICS) + _DIFF_COLUMNS


def load_compare_reports(compare_dir: str) -> list[dict]:
    """compare_dir の YYYY-Www.json を週の順に読む（読めないものは飛ばす）。"""
    reports: list[dict] = []
    for path in sorted(glob.glob(os.path.join(compare_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning("比べの JSON を読めず飛ばす: %s (%s)", path, e)
            continue
        if not isinstance(report, dict):
            logger.warning("比べの JSON の形が違うため飛ばす: %s", path)
            continue
        report.setdefault("week", os.path.splitext(os.path.basename(path))[0])
        reports.append(report)
    return reports


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _diff(value: float | None, base: float | None) -> float | None:
    if value is None or base is None:
        return None
    return value - base


def _market_rows(week: str, market: str, section: dict) -> list[dict]:
    forms = section.get("forms")
    if not isinstance(forms, list):
        logger.warning("比べの JSON %s/%s に forms が無いため飛ばす", week, market)
        return []
    period = section.get("period")
    period = period if isinstance(period, dict) else {}
    base = {
        "week": week,
        "market": market,
        "period_start": period.get("start"),
        "period_end": period.get("end"),
        "symbols": _number(section.get("symbols")),
        "symbols_failed": _number(section.get("symbols_failed")),
    }
    entries = [form for form in forms if isinstance(form, dict)]
    fixed = next((form for form in entries if form.get("form") == FIXED_FORM), None)
    rows: list[dict] = []
    for form in entries:
        row: dict[str, Any] = dict(base)
        row["form"] = str(form.get("form", ""))
        row["k"] = _number(form.get("k"))
        for metric in FORM_METRICS:
            row[metric] = _number(form.get(metric))
        for metric, column in zip(DIFF_METRICS, _DIFF_COLUMNS):
            fixed_value = None if fixed is None else _number(fixed.get(metric))
            row[column] = _diff(row[metric], fixed_value)
        rows.append(row)
    return rows


def _report_rows(report: dict) -> list[dict]:
    week = str(report.get("week", ""))
    markets = report.get("markets")
    if not isinstance(markets, dict):
        logger.warning("比べの JSON %s に markets が無いため飛ばす", week)
        return []
    rows: list[dict] = []
    for market, section in sorted(markets.items()):
        if isinstance(section, dict):
            rows.extend(_market_rows(week, str(market), section))
    return rows


def build_compare_trend(reports: list[dict]) -> pd.DataFrame:
    """週・市場・形・k ごとの値と fixed との差の表を返す（週・市場の順、形は JSON の順）。"""
    rows = [row for report in reports for row in _report_rows(report)]
    df = pd.DataFrame(rows, columns=TREND_COLUMNS)
    numeric = ["k", "symbols", "symbols_failed"] + list(FORM_METRICS) + _DIFF_COLUMNS
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
    return df.sort_values(["week", "market"], kind="stable").reset_index(drop=True)


def load_compare_trend(compare_dir: str) -> pd.DataFrame:
    """compare_dir の JSON を読み、fixed との差つきの表を返す。"""
    return build_compare_trend(load_compare_reports(compare_dir))


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.4f}"
    return str(value)


def format_compare_markdown(df: pd.DataFrame) -> str:
    """表を Markdown の表の文字列にする（行が無ければ「データなし」）。"""
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
