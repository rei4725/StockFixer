"""results/optimal_params.json を読み返し、市場ごとの集計の表を作る（#1950）。

週の最適化（backtest/optimizer/persistence.py）は銘柄ごとに1行を上書きで書き、発注の時は
1銘柄の行を読むだけで、ファイル全体をまとめて読む道が無かった。ここでは読むだけで、
市場ごとに1行の表（行の数・timestamp の古さ・num_trades の区分・threshold・
stop_loss_pct の null・position_sizing）を作る。本番のジョブには組み込まない。

- 古い行は、timestamp の日付が今日の日付より stale_days 日を超えて前の行
- 読めないファイル・形の違う行は警告を出して飛ばす
"""

import json
import math
import os
from datetime import datetime
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_STALE_DAYS = 14
ROW_COLUMNS = [
    "key",
    "market",
    "symbol",
    "timestamp",
    "threshold",
    "stop_loss_pct",
    "position_sizing",
    "num_trades",
]
SUMMARY_COLUMNS = [
    "market",
    "rows",
    "oldest",
    "newest",
    "stale",
    "trades_0",
    "trades_1_2",
    "trades_3_9",
    "trades_10_plus",
    "threshold_min",
    "threshold_median",
    "threshold_max",
    "stop_loss_null",
    "position_sizing",
]
STALE_COLUMNS = ["key", "market", "symbol", "timestamp", "age_days", "num_trades", "threshold"]


def load_optimal_params(path: str) -> dict | None:
    """path の JSON を読む。無い・読めない・形の違うときは None。"""
    if not os.path.exists(path):
        logger.info("最適パラメータの JSON が無い: %s", path)
        return None
    try:
        with open(path, encoding="utf-8") as f:
            params = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning("最適パラメータの JSON を読めない: %s (%s)", path, e)
        return None
    if not isinstance(params, dict):
        logger.warning("最適パラメータの JSON の形が違う: %s", path)
        return None
    return params


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=None)


def _param_row(key: str, entry: Any) -> dict | None:
    if not isinstance(entry, dict):
        return None
    market = entry.get("market")
    timestamp = _parse_timestamp(entry.get("timestamp"))
    threshold = _number(entry.get("threshold"))
    metrics = entry.get("metrics")
    num_trades = _number(metrics.get("num_trades")) if isinstance(metrics, dict) else None
    if not isinstance(market, str) or timestamp is None or threshold is None:
        return None
    if num_trades is None:
        return None
    sizing = entry.get("position_sizing")
    return {
        "key": key,
        "market": market,
        "symbol": str(entry.get("symbol", "")),
        "timestamp": timestamp,
        "threshold": threshold,
        "stop_loss_pct": _number(entry.get("stop_loss_pct")),
        "position_sizing": sizing if isinstance(sizing, str) else "unknown",
        "num_trades": int(num_trades),
    }


def build_param_rows(params: dict) -> pd.DataFrame:
    """銘柄ごとの行の表を返す（形の違う行は警告を出して飛ばす）。"""
    rows: list[dict] = []
    for key, entry in params.items():
        row = _param_row(str(key), entry)
        if row is None:
            logger.warning("最適パラメータの行の形が違うため飛ばす: %s", key)
            continue
        rows.append(row)
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


def _age_days(rows: pd.DataFrame, now: datetime) -> pd.Series:
    today = now.date()
    return rows["timestamp"].map(lambda ts: (today - ts.date()).days)


def _sizing_counts(values: pd.Series) -> str:
    counts = values.value_counts()
    return ";".join(f"{name}:{int(counts[name])}" for name in sorted(counts.index))


def _market_summary(market: str, group: pd.DataFrame, ages: pd.Series, stale_days: int) -> dict:
    trades = group["num_trades"]
    thresholds = group["threshold"]
    return {
        "market": market,
        "rows": len(group),
        "oldest": group["timestamp"].min().date().isoformat(),
        "newest": group["timestamp"].max().date().isoformat(),
        "stale": int((ages > stale_days).sum()),
        "trades_0": int((trades == 0).sum()),
        "trades_1_2": int(trades.between(1, 2).sum()),
        "trades_3_9": int(trades.between(3, 9).sum()),
        "trades_10_plus": int((trades >= 10).sum()),
        "threshold_min": float(thresholds.min()),
        "threshold_median": float(thresholds.median()),
        "threshold_max": float(thresholds.max()),
        "stop_loss_null": int(group["stop_loss_pct"].isna().sum()),
        "position_sizing": _sizing_counts(group["position_sizing"]),
    }


def summarize_by_market(
    rows: pd.DataFrame, now: datetime | None = None, stale_days: int = DEFAULT_STALE_DAYS
) -> pd.DataFrame:
    """市場ごとに1行の集計の表を返す（市場の名の順）。"""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    ages = _age_days(rows, now or datetime.now())
    summaries = [
        _market_summary(str(market), group, ages.loc[group.index], stale_days)
        for market, group in rows.groupby("market", sort=True)
    ]
    return pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)


def stale_rows(
    rows: pd.DataFrame, now: datetime | None = None, stale_days: int = DEFAULT_STALE_DAYS
) -> pd.DataFrame:
    """stale_days 日より古い銘柄の一覧を返す（古い順、同じ日なら key の順）。"""
    if rows.empty:
        return pd.DataFrame(columns=STALE_COLUMNS)
    df = rows.copy()
    df["age_days"] = _age_days(df, now or datetime.now())
    df = df[df["age_days"] > stale_days]
    df = df.sort_values(["timestamp", "key"], kind="stable").reset_index(drop=True)
    return df[STALE_COLUMNS]


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.4f}"
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return str(value)


def format_summary_markdown(df: pd.DataFrame) -> str:
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
