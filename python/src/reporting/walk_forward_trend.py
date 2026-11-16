"""週の Walk-Forward の wf_summary_*.csv を全部読み、回と市場ごとの流れの表を作る（#1926）。

書く側（backtest/walk_forward_report.py）は回ごとに wf_summary_<YYYYMMDD_HHMMSS>.csv を書くが、
読む側は直前・最新の1つしか読まず、全体の成績が週ごとにどう動いたかを読み返す道が無かった。
ここでは読むだけで、回（ファイル名の時刻）と市場（ALL と各 market）ごとに1行の表を作る。
本番のジョブには組み込まない。

- 読むのは wf_summary_YYYYMMDD_HHMMSS.csv だけで、名前の時刻の順に並べる
- 読めないファイル・market の列の無いファイルは警告を出して飛ばす
- 無い成績の列（古い形）は欠け（NaN）にする
- error の列に空でない値がある銘柄を error の数に数える。平均と中央値は欠けを除いて計算する
"""

import os
import re
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_FILE_PATTERN = re.compile(r"^wf_summary_(\d{8}_\d{6})\.csv$")

ALL_MARKETS = "ALL"
METRICS: tuple[str, ...] = (
    "total_return",
    "sharpe_ratio",
    "max_drawdown",
    "cost_impact_return",
    "win_rate",
    "num_trades",
)
STAT_COLUMNS: tuple[str, ...] = tuple(f"{m}_{s}" for m in METRICS for s in ("mean", "median"))
COLUMNS: tuple[str, ...] = ("run", "market", "n_symbols", "n_errors") + STAT_COLUMNS


def load_walk_forward_summaries(report_dir: str) -> list[tuple[str, pd.DataFrame]]:
    """report_dir の wf_summary_*.csv を名前の時刻の順に読み、(回, 表) の列を返す。"""
    try:
        names = os.listdir(report_dir)
    except OSError as e:
        logger.warning("Walk-Forward の報告の置き場を読めない: %s (%s)", report_dir, e)
        return []
    runs: list[tuple[str, pd.DataFrame]] = []
    matched = sorted((m.group(1), name) for name in names if (m := _FILE_PATTERN.match(name)))
    for run, name in matched:
        path = os.path.join(report_dir, name)
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as e:
            logger.warning("wf_summary を読めず飛ばす: %s (%s)", path, e)
            continue
        if "market" not in df.columns:
            logger.warning("wf_summary に market の列が無いため飛ばす: %s", path)
            continue
        runs.append((run, df))
    return runs


def _error_mask(df: pd.DataFrame) -> pd.Series:
    if "error" not in df.columns:
        return pd.Series(False, index=df.index)
    errors = df["error"]
    return errors.notna() & (errors.astype(str).str.strip() != "")


def _stats_row(run: str, market: str, df: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run": run,
        "market": market,
        "n_symbols": int(len(df)),
        "n_errors": int(_error_mask(df).sum()),
    }
    for metric in METRICS:
        if metric in df.columns:
            values = pd.to_numeric(df[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_median"] = float(values.median())
        else:
            row[f"{metric}_mean"] = float("nan")
            row[f"{metric}_median"] = float("nan")
    return row


def build_walk_forward_trend(runs: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """回ごとに ALL と各市場の行を作り、1つの表にして返す。"""
    rows: list[dict[str, Any]] = []
    for run, df in runs:
        rows.append(_stats_row(run, ALL_MARKETS, df))
        markets = sorted(str(m) for m in df["market"].dropna().unique())
        for market in markets:
            rows.append(_stats_row(run, market, df[df["market"].astype(str) == market]))
    return pd.DataFrame(rows, columns=list(COLUMNS)).reset_index(drop=True)


def load_walk_forward_trend(report_dir: str) -> pd.DataFrame:
    """report_dir の wf_summary_*.csv を読み、回と市場ごとの表を返す。"""
    return build_walk_forward_trend(load_walk_forward_summaries(report_dir))


def _cell(value: Any) -> str:
    if value is None or value is pd.NA:
        return "null"
    if isinstance(value, float):
        return "null" if value != value else f"{value:.4g}"
    return str(value)


def format_walk_forward_trend_markdown(df: pd.DataFrame) -> str:
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


def format_walk_forward_trend_csv(df: pd.DataFrame) -> str:
    """表を CSV の文字列にする（欠けは null と書き、0 と分ける）。"""
    return df.to_csv(index=False, na_rep="null")
