"""results/monthly の月次レポートの Markdown を読み返し、月ごとに1行の流れの表を作る（#2025）。

書く側（reporting/monthly.py の save_monthly_report_to_file）は1か月1ファイルの Markdown で、
月をまたいだ流れを見るにはファイルを1つずつ開くしかなかった。ここでは読むだけで、
KPI・週次 Hit Rate・paper/real 乖離・ドリフト件数を月ごとに1行の表にする。
本番のジョブには組み込まない。

- 月によって書式が少し違う（Slippage の説明の行の有無、基準の物差しの節の有無）。
  知っている節の表の行と「閾値超過銘柄数」の行だけを拾い、ほかの行は読まない
- `N/A` と無い行は空にする
- 読めないファイル・月次レポートの形でないファイルは警告を出して飛ばす
"""

import glob
import os
import re

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_TITLE_PREFIX = "# StockFixer 月次レポート"
_KPI_SECTION = "## KPI サマリー"
_WEEKLY_SECTION = "## 週次 Hit Rate トレンド"
_DIFF_SECTION = "## paper/real 乖離"
_DRIFT_SECTION = "## ドリフト状況"
_DRIFT_COUNT_PREFIX = "閾値超過銘柄数:"
_PAST_AVG_LABEL = re.compile(r"^過去 \d+ 週平均 Hit Rate$")

# (節, 表の行の名前) → 列の名前
_TABLE_FIELDS: dict[tuple[str, str], str] = {
    (_KPI_SECTION, "Net Return"): "net_return",
    (_KPI_SECTION, "Max Drawdown"): "max_drawdown",
    (_KPI_SECTION, "Sharpe Ratio"): "sharpe_ratio",
    (_KPI_SECTION, "Hit Rate"): "hit_rate",
    (_KPI_SECTION, "Avg Slippage"): "avg_slippage",
    (_KPI_SECTION, "集計銘柄数"): "symbol_count",
    (_KPI_SECTION, "WF スナップショット"): "wf_snapshot",
    (_WEEKLY_SECTION, "当週 Hit Rate"): "week_hit_rate",
    (_WEEKLY_SECTION, "ドリフト検知"): "drift_detected",
    (_DIFF_SECTION, "追跡件数"): "tracked_count",
    (_DIFF_SECTION, "比較可能件数"): "comparable_count",
    (_DIFF_SECTION, "Paper Slippage"): "paper_slippage",
}

TREND_COLUMNS: list[str] = [
    "month",
    "net_return",
    "max_drawdown",
    "sharpe_ratio",
    "hit_rate",
    "avg_slippage",
    "symbol_count",
    "wf_snapshot",
    "week_hit_rate",
    "past_avg_hit_rate",
    "drift_detected",
    "tracked_count",
    "comparable_count",
    "paper_slippage",
    "drift_symbol_count",
]

_MONTH_PATTERN = re.compile(r"^(\d{4}-\d{2})_report\.md$")


def _clean(value: str) -> str:
    """セルの値を整える（N/A は空、太字の印は外す）。"""
    text = value.strip().replace("**", "").strip()
    if text == "N/A":
        return ""
    if text.startswith("YES"):
        return "YES"
    return text


def _section_of(line: str) -> str | None:
    if not line.startswith("## "):
        return None
    for section in (_KPI_SECTION, _WEEKLY_SECTION, _DIFF_SECTION, _DRIFT_SECTION):
        if line.startswith(section):
            return section
    return ""


def parse_monthly_report(text: str, month: str) -> dict[str, str] | None:
    """月次レポートの Markdown から1行分の値を取り出す（形が違えば None）。"""
    lines = text.splitlines()
    if not lines or not lines[0].startswith(_TITLE_PREFIX):
        return None
    row = {column: "" for column in TREND_COLUMNS}
    row["month"] = month
    section = ""
    for raw in lines:
        line = raw.strip()
        next_section = _section_of(line)
        if next_section is not None:
            section = next_section
            continue
        if section == _DRIFT_SECTION and line.startswith(_DRIFT_COUNT_PREFIX):
            row["drift_symbol_count"] = _clean(line[len(_DRIFT_COUNT_PREFIX) :])
            continue
        if not (line.startswith("|") and line.endswith("|")):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 2:
            continue
        label, value = cells
        if section == _WEEKLY_SECTION and _PAST_AVG_LABEL.match(label):
            row["past_avg_hit_rate"] = _clean(value)
            continue
        column = _TABLE_FIELDS.get((section, label))
        if column is not None:
            row[column] = _clean(value)
    return row


def load_monthly_reports(monthly_dir: str) -> list[dict[str, str]]:
    """monthly_dir の YYYY-MM_report.md を月の順に読む（読めないものは飛ばす）。"""
    rows: list[dict[str, str]] = []
    for path in sorted(glob.glob(os.path.join(monthly_dir, "*_report.md"))):
        match = _MONTH_PATTERN.match(os.path.basename(path))
        if match is None:
            logger.warning("月次レポートの名前の形が違うため飛ばす: %s", path)
            continue
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("月次レポートを読めず飛ばす: %s (%s)", path, e)
            continue
        row = parse_monthly_report(text, match.group(1))
        if row is None:
            logger.warning("月次レポートの形が違うため飛ばす: %s", path)
            continue
        rows.append(row)
    return rows


def build_monthly_report_trend(rows: list[dict[str, str]]) -> pd.DataFrame:
    """月ごとの行を月の順の表にする。"""
    df = pd.DataFrame(rows, columns=TREND_COLUMNS)
    return df.sort_values("month", kind="stable").reset_index(drop=True)


def load_monthly_report_trend(monthly_dir: str) -> pd.DataFrame:
    """monthly_dir の月次レポートを読み、月ごとの表を返す。"""
    return build_monthly_report_trend(load_monthly_reports(monthly_dir))


def format_monthly_trend_markdown(df: pd.DataFrame) -> str:
    """表を Markdown の表の文字列にする（行が無ければ「データなし」）。"""
    if df.empty:
        return "データなし"
    columns = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in values) + " |")
    return "\n".join(lines)
