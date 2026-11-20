"""results/paper の週ごとの紙の売買の成績の JSON を読み返し、週・市場ごとの表を作る（#1862）。

書く側（trading/paper_equity.py）の JSON は1週1ファイルで、週をまたいだ流れを見るには
ファイルを1つずつ開くしかなかった。ここでは読むだけで、週・市場ごとの1行の表と、
週の騰落率と超過の騰落率の複利の累計を作る。本番のジョブには組み込まない。

- #1792 より前の古い JSON には exposure・excess_basis・excess_return_unscaled が無い。
  その欄は空にする（古い excess_return は全額のベンチマークとの差）
- 週の騰落率の累計は市場ごとに取る。超過の騰落率の累計は、計算の印（excess_basis。
  古い週は印なし）が違う週をまたがないよう、市場と印の組ごとに分けて取る
- 値の無い（None の）週は累計に掛けず、その行の累計の欄は空にする
- 読めないファイル・形の違う JSON は警告を出して飛ばす
- 右端に、約定の値段で評価した銘柄と週末の終値の無い銘柄の数を置く（#2062）。
  鍵の無い古い週は空にし、累計の計算には使わない
"""

import glob
import json
import math
import os
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

RETURN_METRICS: tuple[str, ...] = (
    "weekly_return",
    "benchmark_return",
    "excess_return",
    "excess_return_unscaled",
    "exposure",
)
_KEY_COLUMNS = ["week", "start", "end", "market"]
_BASE_COLUMNS = (
    _KEY_COLUMNS
    + ["benchmark_ticker"]
    + list(RETURN_METRICS)
    + ["excess_basis", "fills", "partial_week"]
)
CUM_WEEKLY_COLUMN = "cum_weekly_return"
CUM_EXCESS_COLUMN = "cum_excess_return"
_NO_BASIS = "__no_basis__"
# 表の右端に置く銘柄の数の列（#2062）。書く側の並びの長さで、鍵の無い古い週は空
COUNT_COLUMNS: dict[str, str] = {
    "valued_at_fill_price_count": "valued_at_fill_price",
    "missing_week_end_close_count": "missing_week_end_close",
}


def load_paper_weekly_reports(paper_dir: str) -> list[dict]:
    """paper_dir の YYYY-Www.json を週の順に読む（読めないものは飛ばす）。"""
    reports: list[dict] = []
    for path in sorted(glob.glob(os.path.join(paper_dir, "*-W*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning("紙の週の JSON を読めず飛ばす: %s (%s)", path, e)
            continue
        if not isinstance(report, dict):
            logger.warning("紙の週の JSON の形が違うため飛ばす: %s", path)
            continue
        report.setdefault("week", os.path.splitext(os.path.basename(path))[0])
        reports.append(report)
    return reports


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _row(report: dict, market: str, section: dict) -> dict:
    partial = section.get("partial_week")
    row: dict[str, Any] = {
        "week": str(report.get("week", "")),
        "start": _text(report.get("start")),
        "end": _text(report.get("end")),
        "market": market,
        "benchmark_ticker": _text(section.get("benchmark_ticker")),
        "excess_basis": _text(section.get("excess_basis")),
        "fills": _number(section.get("fills")),
        "partial_week": partial if isinstance(partial, bool) else None,
    }
    for metric in RETURN_METRICS:
        row[metric] = _number(section.get(metric))
    for column, key in COUNT_COLUMNS.items():
        symbols = section.get(key)
        row[column] = float(len(symbols)) if isinstance(symbols, list) else None
    return row


def _report_rows(report: dict) -> list[dict]:
    markets = report.get("markets")
    if not isinstance(markets, dict):
        logger.warning("紙の週の JSON %s に markets が無いため飛ばす", report.get("week"))
        return []
    return [
        _row(report, str(market), section)
        for market, section in sorted(markets.items())
        if isinstance(section, dict)
    ]


def _compound(values: pd.Series) -> pd.Series:
    """値のある行だけを掛け合わせた複利の累計（値の無い行は NaN）。"""
    running = (1.0 + values.fillna(0.0)).cumprod() - 1.0
    return running.where(values.notna())


def build_paper_weekly_trend(reports: list[dict]) -> pd.DataFrame:
    """週・市場ごとの成績の表に、週の騰落率と超過の騰落率の複利の累計の列を足して返す。"""
    rows = [row for report in reports for row in _report_rows(report)]
    counts = pd.DataFrame(rows, columns=list(COUNT_COLUMNS))
    df = pd.DataFrame(rows, columns=_BASE_COLUMNS)
    for column in list(RETURN_METRICS) + ["fills"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
    for column in COUNT_COLUMNS:
        counts[column] = pd.to_numeric(counts[column], errors="coerce").astype(float)
    order = df.sort_values(["week", "market"], kind="stable").index
    df = df.loc[order].reset_index(drop=True)
    counts = counts.loc[order].reset_index(drop=True)
    df[CUM_WEEKLY_COLUMN] = df.groupby("market", sort=False)["weekly_return"].transform(_compound)
    basis_key = df["excess_basis"].fillna(_NO_BASIS)
    df[CUM_EXCESS_COLUMN] = df.groupby([df["market"], basis_key], sort=False)[
        "excess_return"
    ].transform(_compound)
    # 銘柄の数は累計の後ろ（右端）に置き、累計の計算には使わない（#2062）
    for column in COUNT_COLUMNS:
        df[column] = counts[column]
    return df


def load_paper_weekly_trend(paper_dir: str) -> pd.DataFrame:
    """paper_dir の JSON を読み、累計つきの表を返す。"""
    return build_paper_weekly_trend(load_paper_weekly_reports(paper_dir))


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.6f}"
    return str(value)


def format_paper_weekly_markdown(df: pd.DataFrame) -> str:
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
