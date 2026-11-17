"""品質の記録 data_quality_log を読み返し、日・市場・check_name・level ごとの流れの表を作る（#1951）。

書く側（market_data/feature_save.py が utils/db/quality_log.insert_quality_log を呼ぶ）は、
品質の結果を1件1行で書き、知らせるのは error と zero_volume だけである。ここでは読むだけで、
日（checked_at の UTC の日付）・市場・check_name・level ごとに、件数と銘柄の数を1行にした
表を作る。本番のジョブには組み込まない。

- 知らない check_name もそのまま行にする
- 銘柄の数は、同じ日・市場・check_name・level に同じ銘柄が何度書かれても1つに数える
"""

import math
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

from src.utils.db.quality_log import QualityLogRecord

_KEY_COLUMNS = ["date", "market", "check_name", "level"]
_COLUMNS = _KEY_COLUMNS + ["count", "symbols"]


def checked_date(checked_at: str) -> str:
    """checked_at（ISO 8601 の文字列）を UTC の日付 YYYY-MM-DD にする（tz の無い値は UTC とみなす）。"""
    value = datetime.fromisoformat(checked_at)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).date().isoformat()


def build_data_quality_trend(records: Iterable[QualityLogRecord]) -> pd.DataFrame:
    """日・市場・check_name・level ごとに count（件数）と symbols（銘柄の数）を数えた表を返す。"""
    counts: dict[tuple[str, str, str, str], int] = {}
    symbols: dict[tuple[str, str, str, str], set[str]] = {}
    for record in records:
        key = (
            checked_date(record.checked_at),
            str(record.market),
            str(record.check_name),
            str(record.level),
        )
        counts[key] = counts.get(key, 0) + 1
        symbols.setdefault(key, set()).add(str(record.symbol))
    rows = [
        dict(zip(_KEY_COLUMNS, key), count=count, symbols=len(symbols[key]))
        for key, count in counts.items()
    ]
    df = pd.DataFrame(rows, columns=_COLUMNS)
    for column in ("count", "symbols"):
        df[column] = df[column].astype(int)
    return df.sort_values(_KEY_COLUMNS, kind="stable").reset_index(drop=True)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.4f}"
    return str(value)


def format_data_quality_trend_markdown(df: pd.DataFrame) -> str:
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
