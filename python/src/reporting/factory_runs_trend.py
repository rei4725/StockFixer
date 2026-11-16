"""戦略ファクトリーの台帳 factory_runs を読み返し、夜ごと・市場ごとの流れの表を作る（#1907）。

書く側（backtest/factory.py）は、候補ごとに gate_passed と gate_reasons（"; " でつないだ
文字列）を factory_runs に書く。ここでは読むだけで、評価の日（evaluated_at の日付）・市場ごとに
評価の数・合格の数・ゲートの理由ごとの落ちた数を1行にした表を作る。本番のジョブには組み込まない。

- 理由の文の形は backtest/factory_gate.py の今の文から取る。知らない形は other に数える
- 1件が複数の理由で落ちたら、各理由に1つずつ数える（同じ理由が1件に2度出ても1つ）
- 合格の行（理由が null）はどの理由にも数えない。不合格なのに理由が null の行は other に数える
"""

import math
from typing import Any, Iterable, Optional

import pandas as pd

from src.utils.db.factory_runs import FactoryRunRecord

REASON_SEPARATOR = "; "
OTHER = "other"
# (理由の名前, 文の頭の形)。上から順に当て、最初に当たった名前に数える。
_REASON_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("num_trades", ("num_trades ",)),
    ("effective_symbols", ("effective_symbols ",)),
    ("dsr", ("dsr ",)),
    ("max_drawdown", ("portfolio_max_drawdown ", "max_drawdown ")),
    # チャンピオンとの Sharpe 差の z が足りない、または日次リターンが無く検定できない
    ("champion_z", ("portfolio_sharpe ", "チャンピオンとの Sharpe 差を検定できない")),
    ("no_champion", ("チャンピオンが無い",)),
)
REASON_CATEGORIES: tuple[str, ...] = tuple(name for name, _ in _REASON_PREFIXES) + (OTHER,)
_KEY_COLUMNS = ["date", "market"]
_REASON_COLUMNS = [f"failed_{name}" for name in REASON_CATEGORIES]
_COLUMNS = _KEY_COLUMNS + ["evaluated", "passed"] + _REASON_COLUMNS


def classify_reason(reason: str) -> str:
    """理由の文1つを REASON_CATEGORIES のどれかに分ける（知らない形は other）。"""
    text = reason.strip()
    for name, prefixes in _REASON_PREFIXES:
        if text.startswith(prefixes):
            return name
    return OTHER


def split_gate_reasons(gate_reasons: Optional[str]) -> list[str]:
    """ "; " でつないだ理由の文字列を、理由の名前の重なりの無い並びにする（null・空は空の並び）。"""
    if not gate_reasons:
        return []
    names: list[str] = []
    for part in gate_reasons.split(REASON_SEPARATOR):
        if not part.strip():
            continue
        name = classify_reason(part)
        if name not in names:
            names.append(name)
    return names


def _row_reasons(record: FactoryRunRecord) -> list[str]:
    if record.gate_passed:
        return []
    return split_gate_reasons(record.gate_reasons) or [OTHER]


def build_factory_runs_trend(records: Iterable[FactoryRunRecord]) -> pd.DataFrame:
    """評価の日・市場ごとに evaluated・passed・failed_<理由> を数えた表を返す。"""
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for record in records:
        key = (record.evaluated_at.date().isoformat(), str(record.market))
        row = counts.setdefault(key, {column: 0 for column in _COLUMNS[2:]})
        row["evaluated"] += 1
        if record.gate_passed:
            row["passed"] += 1
        for name in _row_reasons(record):
            row[f"failed_{name}"] += 1
    rows = [dict(date=day, market=market, **row) for (day, market), row in counts.items()]
    df = pd.DataFrame(rows, columns=_COLUMNS)
    for column in _COLUMNS[2:]:
        df[column] = df[column].astype(int)
    return df.sort_values(_KEY_COLUMNS, kind="stable").reset_index(drop=True)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.4f}"
    return str(value)


def format_factory_runs_trend_markdown(df: pd.DataFrame) -> str:
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
