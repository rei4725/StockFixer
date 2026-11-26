"""16:30 のルールの道の記録の zero_qty_symbols を日・銘柄ごとに読み返す表を作る（#2189）。

書く側（trading/execution/gate_record.py）は 16:30 の記録 results/orders/rule/<日>.json に、株数0で
飛ばした新規買いの銘柄ごとの行 zero_qty_symbols（symbol・reason・price・lot_cost・sized_qty・
budget）を書く（#1997）。ここでは読むだけで、日・銘柄ごとに1行の表と、銘柄ごとに落ちた日数・
最後の理由・最後の lot_cost と budget の表を作る。本番のジョブには組み込まない。

- 読むのは results/orders/rule/*.json だけ
- 鍵の無い古い記録は飛ばす（行を作らない。鍵のある日数 recorded_days にも数えない）
- 鍵が空のリストの記録は、その日に株数0で落ちた銘柄が無かった日として recorded_days に数える
- 辞書でない行・symbol の無い行は飛ばす。数でない値（null・文字・真偽）は null にする
- 読めないファイル・形の違う JSON は警告を出して飛ばす（order_sector_cap_trend.py の読み方にそろえる）
"""

import glob
import json
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

ZERO_QTY_KEY = "zero_qty_symbols"
DAILY_COLUMNS: tuple[str, ...] = (
    "date",
    "symbol",
    "reason",
    "price",
    "lot_cost",
    "budget",
    "sized_qty",
)
SUMMARY_COLUMNS: tuple[str, ...] = (
    "symbol",
    "days",
    "recorded_days",
    "reason_days",
    "last_reason",
    "last_lot_cost",
    "last_budget",
    "last_date",
)


@dataclass(frozen=True)
class ZeroQtyRow:
    """株数0で落ちた新規買いの1行。数でない値は None。"""

    symbol: str
    reason: str
    price: float | None
    lot_cost: float | None
    budget: float | None
    sized_qty: int | None


@dataclass(frozen=True)
class ZeroQtyDay:
    """鍵のある1日分の記録。rows は読めた行だけ（空なら0株で落ちた銘柄の無い日）。"""

    date: str
    rows: tuple[ZeroQtyRow, ...]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _int(value: Any) -> int | None:
    number = _number(value)
    return None if number is None else int(number)


def _load_records(rule_dir: str) -> list[dict]:
    records: list[dict] = []
    for path in sorted(glob.glob(os.path.join(rule_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning("ルールの道の記録の JSON を読めず飛ばす: %s (%s)", path, e)
            continue
        if not isinstance(record, dict):
            logger.warning("ルールの道の記録の JSON の形が違うため飛ばす: %s", path)
            continue
        record.setdefault("date", os.path.splitext(os.path.basename(path))[0])
        records.append(record)
    return records


def _to_row(item: Any) -> ZeroQtyRow | None:
    if not isinstance(item, dict):
        return None
    symbol = item.get("symbol")
    if symbol is None or str(symbol) == "":
        return None
    reason = item.get("reason")
    return ZeroQtyRow(
        symbol=str(symbol),
        reason="" if reason is None else str(reason),
        price=_number(item.get("price")),
        lot_cost=_number(item.get("lot_cost")),
        budget=_number(item.get("budget")),
        sized_qty=_int(item.get("sized_qty")),
    )


def to_zero_qty_days(records: list[dict]) -> list[ZeroQtyDay]:
    """記録のうち zero_qty_symbols がリストのものだけを日の順に返す（鍵の無い記録は飛ばす）。"""
    days: list[ZeroQtyDay] = []
    for record in records:
        items = record.get(ZERO_QTY_KEY)
        if not isinstance(items, list):
            continue
        rows = tuple(row for row in (_to_row(item) for item in items) if row is not None)
        days.append(ZeroQtyDay(date=str(record.get("date", "")), rows=rows))
    return sorted(days, key=lambda d: d.date)


def load_zero_qty_days(rule_dir: str) -> list[ZeroQtyDay]:
    """rule_dir の 16:30 の記録を読み、鍵のある日の記録を返す。"""
    return to_zero_qty_days(_load_records(rule_dir))


def tail_days(days: list[ZeroQtyDay], n: int | None) -> list[ZeroQtyDay]:
    """鍵のある日のうち直近 n 日だけを返す（n が無いか 0 以下なら全部）。"""
    if not n or n <= 0:
        return list(days)
    return list(days[-n:])


def build_zero_qty_daily(days: list[ZeroQtyDay]) -> pd.DataFrame:
    """日・銘柄ごとに1行の表。日の中は銘柄の順。"""
    rows = [
        {
            "date": day.date,
            "symbol": row.symbol,
            "reason": row.reason,
            "price": row.price,
            "lot_cost": row.lot_cost,
            "budget": row.budget,
            "sized_qty": row.sized_qty,
        }
        for day in days
        for row in day.rows
    ]
    df = pd.DataFrame(rows, columns=list(DAILY_COLUMNS))
    for column in ("price", "lot_cost", "budget"):
        df[column] = df[column].astype("Float64")
    df["sized_qty"] = df["sized_qty"].astype("Int64")
    return df.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)


def build_zero_qty_summary(days: list[ZeroQtyDay]) -> pd.DataFrame:
    """銘柄ごとの落ちた日数 days・鍵のある日数 recorded_days・理由ごとの日数と最後の行の表。

    reason_days は「理由=日数」を「;」でつないだ文字列。同じ日に同じ銘柄が2行あっても1日に数える。
    last_* はその銘柄が最後に出た日の、その日の最後の行の値。
    """
    hit_days: dict[str, set[str]] = {}
    reason_days: dict[str, dict[str, set[str]]] = {}
    last: dict[str, tuple[str, ZeroQtyRow]] = {}
    for day in days:
        for row in day.rows:
            hit_days.setdefault(row.symbol, set()).add(day.date)
            reason_days.setdefault(row.symbol, {}).setdefault(row.reason, set()).add(day.date)
            last[row.symbol] = (day.date, row)
    rows = []
    for symbol, dates in hit_days.items():
        last_date, last_row = last[symbol]
        by_reason = reason_days[symbol]
        rows.append(
            {
                "symbol": symbol,
                "days": len(dates),
                "recorded_days": len(days),
                "reason_days": ";".join(
                    f"{reason}={len(by_reason[reason])}" for reason in sorted(by_reason)
                ),
                "last_reason": last_row.reason,
                "last_lot_cost": last_row.lot_cost,
                "last_budget": last_row.budget,
                "last_date": last_date,
            }
        )
    df = pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
    for column in ("days", "recorded_days"):
        df[column] = df[column].astype("Int64")
    for column in ("last_lot_cost", "last_budget"):
        df[column] = df[column].astype("Float64")
    return df.sort_values(["days", "symbol"], ascending=[False, True], kind="stable").reset_index(
        drop=True
    )


def _cell(value: Any) -> str:
    if value is None or value is pd.NA:
        return "null"
    return str(value)


def format_zero_qty_markdown(df: pd.DataFrame) -> str:
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


def format_zero_qty_csv(df: pd.DataFrame) -> str:
    """表を CSV の文字列にする。"""
    return df.to_csv(index=False, na_rep="null")
