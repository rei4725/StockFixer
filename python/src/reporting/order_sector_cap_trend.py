"""8:50 の発注の記録の sector_cap_by_sector を日・業種ごとに読み返す表を作る（#2043）。

書く側（trading/execution/gate_record.py）は 8:50 の記録 results/orders/<日>.json に、業種の上限で
落とした新規買いの候補の業種ごとの数 sector_cap_by_sector を書く（#1908。和は skip_reasons の
sector_cap）。ここでは読むだけで、日・業種ごとに1行の表と、業種ごとの合計・出た日数の表を作る。
本番のジョブには組み込まない。

- 読むのは results/orders/*.json（8:50）だけ。16:30 の orders/rule は鍵が無いので読まない
- 鍵の無い古い記録は飛ばす（行を作らない。0 の日とは分け、鍵のある日数 recorded_days にも数えない）
- 鍵が空の {} の記録は、その日に業種の上限で落とした候補が無かった日として recorded_days に数える
- 数でない値（null・文字・真偽）は飛ばす
- 読めないファイル・形の違う JSON は警告を出して飛ばす（order_gate_trend.py の読み方にそろえる）
"""

import glob
import json
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

SECTOR_CAP_KEY = "sector_cap_by_sector"
DAILY_COLUMNS: tuple[str, ...] = ("date", "sector", "dropped")
SUMMARY_COLUMNS: tuple[str, ...] = ("sector", "dropped", "days", "recorded_days")


@dataclass(frozen=True)
class SectorCapDay:
    """鍵のある1日分の記録。by_sector は業種ごとの落とした数（数でない値は除いたもの）。"""

    date: str
    by_sector: dict[str, int]


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _load_records(orders_dir: str) -> list[dict]:
    records: list[dict] = []
    for path in sorted(glob.glob(os.path.join(orders_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning("発注の記録の JSON を読めず飛ばす: %s (%s)", path, e)
            continue
        if not isinstance(record, dict):
            logger.warning("発注の記録の JSON の形が違うため飛ばす: %s", path)
            continue
        record.setdefault("date", os.path.splitext(os.path.basename(path))[0])
        records.append(record)
    return records


def to_sector_cap_days(records: list[dict]) -> list[SectorCapDay]:
    """記録のうち sector_cap_by_sector が辞書のものだけを日の順に返す（鍵の無い記録は飛ばす）。"""
    days: list[SectorCapDay] = []
    for record in records:
        by_sector = record.get(SECTOR_CAP_KEY)
        if not isinstance(by_sector, dict):
            continue
        counts: dict[str, int] = {}
        for sector, value in by_sector.items():
            number = _count(value)
            if number is not None:
                counts[str(sector)] = number
        days.append(SectorCapDay(date=str(record.get("date", "")), by_sector=counts))
    return sorted(days, key=lambda d: d.date)


def load_sector_cap_days(orders_dir: str) -> list[SectorCapDay]:
    """orders_dir の 8:50 の記録を読み、鍵のある日の記録を返す（orders_dir/rule は読まない）。"""
    return to_sector_cap_days(_load_records(orders_dir))


def tail_days(days: list[SectorCapDay], n: int | None) -> list[SectorCapDay]:
    """鍵のある日のうち直近 n 日だけを返す（n が無いか 0 以下なら全部）。"""
    if not n or n <= 0:
        return list(days)
    return list(days[-n:])


def build_sector_cap_daily(days: list[SectorCapDay]) -> pd.DataFrame:
    """日・業種ごとに1行（date・sector・dropped）の表。日の中は落とした数の多い順。"""
    rows = [
        {"date": day.date, "sector": sector, "dropped": dropped}
        for day in days
        for sector, dropped in day.by_sector.items()
    ]
    df = pd.DataFrame(rows, columns=list(DAILY_COLUMNS))
    df["dropped"] = df["dropped"].astype("Int64")
    df = df.sort_values(["dropped", "sector"], ascending=[False, True], kind="stable")
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def build_sector_cap_summary(days: list[SectorCapDay]) -> pd.DataFrame:
    """業種ごとの合計 dropped・落とした日数 days（1件以上の日）・鍵のある日数 recorded_days の表。"""
    totals: dict[str, int] = {}
    hit_days: dict[str, int] = {}
    for day in days:
        for sector, dropped in day.by_sector.items():
            totals[sector] = totals.get(sector, 0) + dropped
            if dropped > 0:
                hit_days[sector] = hit_days.get(sector, 0) + 1
    rows = [
        {
            "sector": sector,
            "dropped": total,
            "days": hit_days.get(sector, 0),
            "recorded_days": len(days),
        }
        for sector, total in totals.items()
    ]
    df = pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
    for column in ("dropped", "days", "recorded_days"):
        df[column] = df[column].astype("Int64")
    return df.sort_values(
        ["dropped", "sector"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def _cell(value: Any) -> str:
    if value is None or value is pd.NA:
        return "null"
    return str(value)


def format_sector_cap_markdown(df: pd.DataFrame) -> str:
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


def format_sector_cap_csv(df: pd.DataFrame) -> str:
    """表を CSV の文字列にする。"""
    return df.to_csv(index=False, na_rep="null")
