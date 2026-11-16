"""results/rule_accuracy の日ごとの JSON の backtest_gap を読み返し、流れの表を作る（#1891）。

書く側（scoring/rule_report.py）は、ルールごとにバックテストの勝率と錨の5日の的中率を
並べた backtest_gap を JSON に書く。ここでは読むだけで、日・市場・ルール・向きごとに
backtest_win_rate・live_hit_rate_h5・gap・n_h5・n_unmatched・n_sell_only・approx・
baseline_rate・edge を1行にした表を作る。本番のジョブには組み込まない。

- 古い形（backtest_gap がルールごとに向きをまとめた1段）の日は、向きを combined として読む
- 新しい形（backtest_gap が buy・sell に分かれる）の日は、backtest_gap_combined を combined とする
- gap は backtest_win_rate - live_hit_rate_h5（どちらかが null なら null）
- 値が null・欄が無い組は NaN（approx は None）のまま残し、0 とは分ける（表では空で出す）
- 読めないファイル・形の違う JSON は警告を出して飛ばす（読み込みは rule_accuracy_trend と同じ）
"""

import math
from typing import Any

import pandas as pd

from src.reporting.rule_accuracy_trend import _number, load_rule_accuracy_reports
from src.utils.logger import get_logger

logger = get_logger(__name__)

GAP_METRICS: tuple[str, ...] = (
    "backtest_win_rate",
    "live_hit_rate_h5",
    "gap",
    "n_h5",
    "n_unmatched",
    "n_sell_only",
    "approx",
    "baseline_rate",
    "edge",
)
_KEY_COLUMNS = ["date", "market", "rule", "direction"]
_COLUMNS = _KEY_COLUMNS + list(GAP_METRICS)
_NUMERIC = [m for m in GAP_METRICS if m != "approx"]
_SIDES = ("buy", "sell")
COMBINED = "combined"


def _gap_row(base: dict[str, Any], rule: str, direction: str, entry: dict) -> dict:
    row = dict(base, rule=str(rule), direction=direction)
    for metric in _NUMERIC:
        if metric != "gap":
            row[metric] = _number(entry.get(metric))
    win, hit = row["backtest_win_rate"], row["live_hit_rate_h5"]
    row["gap"] = None if win is None or hit is None else win - hit
    approx = entry.get("approx")
    row["approx"] = approx if isinstance(approx, bool) else None
    return row


def _is_by_side(entry: dict) -> bool:
    return any(isinstance(entry.get(side), dict) for side in _SIDES)


def _report_gap_rows(report: dict, market: str | None) -> list[dict]:
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
    gap = report.get("backtest_gap")
    if not isinstance(gap, dict):
        logger.warning("ルールの採点の JSON %s に backtest_gap が無いため飛ばす", day)
        return []
    combined = report.get("backtest_gap_combined")
    base = {"date": day, "market": report_market}
    rows: list[dict] = []
    for rule, entry in gap.items():
        if not isinstance(entry, dict):
            continue
        if _is_by_side(entry):
            for side in _SIDES:
                if isinstance(entry.get(side), dict):
                    rows.append(_gap_row(base, rule, side, entry[side]))
        else:
            rows.append(_gap_row(base, rule, COMBINED, entry))
    if isinstance(combined, dict):
        for rule, entry in combined.items():
            if isinstance(entry, dict) and not _is_by_side(entry):
                rows.append(_gap_row(base, rule, COMBINED, entry))
    return rows


def build_rule_backtest_gap_trend(reports: list[dict], market: str | None = None) -> pd.DataFrame:
    """日・市場・ルール・向き（buy・sell・combined）ごとの GAP_METRICS の表を返す。

    market を渡すと、その市場の JSON だけを使い、違う市場の JSON は警告を出して飛ばす。
    """
    rows = [row for report in reports for row in _report_gap_rows(report, market)]
    df = pd.DataFrame(rows, columns=_COLUMNS)
    for column in _NUMERIC:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
    df["approx"] = df["approx"].astype(object)
    df = df.drop_duplicates(subset=_KEY_COLUMNS, keep="first")
    return df.sort_values(_KEY_COLUMNS, kind="stable").reset_index(drop=True)


def filter_gap_trend(
    df: pd.DataFrame, rule: str | None = None, direction: str | None = None
) -> pd.DataFrame:
    """渡した条件の行だけに絞る（None の条件は絞らない）。"""
    out = df
    for column, value in (("rule", rule), ("direction", direction)):
        if value is not None:
            out = out[out[column] == str(value)]
    return out.reset_index(drop=True)


def load_rule_backtest_gap_trend(accuracy_dir: str, market: str | None = None) -> pd.DataFrame:
    """accuracy_dir の JSON を読み、表を返す。"""
    return build_rule_backtest_gap_trend(load_rule_accuracy_reports(accuracy_dir), market)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.4f}"
    return str(value)


def format_gap_trend_markdown(df: pd.DataFrame) -> str:
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
