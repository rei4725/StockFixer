"""8:50 の発注の要約 order_run_summary を読み返し、日・市場ごとの流れの表を作る（#1981）。

書く側（trading/execution/runner.py が infrastructure/persistence/order_run_repository の
PostgresOrderRunSink.save を呼ぶ）は、発注の回ごとに1行を書くだけで、売買の金額 total_turnover を
日をまたいで読み返す道が無かった。results/orders の日ごとの JSON（reporting/order_gate_trend.py）には
total_turnover が無く、results/orders の始まる前の回は表にしか残っていない。ここでは読むだけで、
日（run_at の UTC の日付）・市場ごとに、回の数・件数・total_turnover の合計を1行にした表を作る。
本番のジョブには組み込まない。

- run_at に tz の無い値は UTC とみなす
- mode は値をそのまま写す。同じ日・市場に違う mode の回があれば、値を並べてカンマでつなぐ
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

_KEY_COLUMNS = ["date", "market"]
COUNT_COLUMNS: tuple[str, ...] = (
    "buy_orders",
    "sell_orders",
    "short_orders",
    "skipped",
    "skipped_min_change",
)
_COLUMNS = _KEY_COLUMNS + ["mode", "runs"] + list(COUNT_COLUMNS) + ["total_turnover"]


@dataclass(frozen=True)
class OrderRunRecord:
    """order_run_summary の1行（読み返す分。#1981）。"""

    run_id: str
    market: str
    mode: str
    run_at: datetime
    buy_orders: int
    sell_orders: int
    short_orders: int
    skipped: int
    skipped_min_change: int
    total_turnover: float
    min_change_ratio: float | None


def run_date(run_at: datetime) -> str:
    """run_at を UTC の日付 YYYY-MM-DD にする（tz の無い値は UTC とみなす）。"""
    if run_at.tzinfo is None:
        run_at = run_at.replace(tzinfo=timezone.utc)
    return run_at.astimezone(timezone.utc).date().isoformat()


def build_order_run_trend(records: Iterable[OrderRunRecord]) -> pd.DataFrame:
    """日・市場ごとに、回の数 runs・件数の合計・total_turnover の合計を1行にした表を返す。"""
    sums: dict[tuple[str, str], dict[str, Any]] = {}
    modes: dict[tuple[str, str], set[str]] = {}
    for record in records:
        key = (run_date(record.run_at), str(record.market))
        row = sums.setdefault(
            key, {"runs": 0, "total_turnover": 0.0, **{c: 0 for c in COUNT_COLUMNS}}
        )
        row["runs"] += 1
        for column in COUNT_COLUMNS:
            row[column] += int(getattr(record, column))
        row["total_turnover"] += float(record.total_turnover)
        modes.setdefault(key, set()).add(str(record.mode))
    rows = [
        dict(zip(_KEY_COLUMNS, key), mode=",".join(sorted(modes[key])), **values)
        for key, values in sums.items()
    ]
    df = pd.DataFrame(rows, columns=_COLUMNS)
    for column in ("runs",) + COUNT_COLUMNS:
        df[column] = df[column].astype(int)
    df["total_turnover"] = df["total_turnover"].astype(float)
    return df.sort_values(_KEY_COLUMNS, kind="stable").reset_index(drop=True)


def format_order_run_markdown(df: pd.DataFrame) -> str:
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


def format_order_run_csv(df: pd.DataFrame) -> str:
    """表を CSV の文字列にする。"""
    return df.to_csv(index=False)
