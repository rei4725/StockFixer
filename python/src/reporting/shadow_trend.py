"""results/accuracy/shadow の週ごとの影の比べの JSON を読み返し、週ごとの1行の表を作る（#1863）。

書く側（prediction/shadow_record.py）の JSON は1週1ファイルで、challenger が勝たない週や
定数の予想で比べを飛ばした週が何週続いたかを見るには、ファイルを1つずつ開くしかなかった。
ここでは読むだけで、本番のジョブには組み込まない。

- 列は記録の欄に、challenger と production の的中率の差（hit_rate_diff）と、
  challenger_wins が false の週・challenger_constant が true の週の続いた数を足す
- 右端に、影の比べとゲートの成否・今週の challenger の数・shadow_error の頭・
  勝ちの決め方（daily_rank_corr と win_rule）の列を足す（#2061）
- 欄の無い古い週はその列を空にする。続いた数は、その欄の値が条件に合う週だけを数え、
  合わない週と欄の無い週で 0 に戻る
- 読めないファイル・形の違う JSON は警告を出して飛ばす
"""

import glob
import json
import math
import os
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

NUMBER_COLUMNS: tuple[str, ...] = (
    "window_days",
    "production_hit_rate",
    "challenger_hit_rate",
    "production_sharpe",
    "challenger_sharpe",
    "n_production",
    "n_challenger",
    "challenger_prediction_std",
    "retry_count",
)
FLAG_COLUMNS: tuple[str, ...] = ("challenger_wins", "challenger_constant", "promoted")
TEXT_COLUMNS: tuple[str, ...] = ("written_at", "gate_skipped_reason")
DIFF_COLUMN = "hit_rate_diff"
NOT_WIN_STREAK = "challenger_not_win_streak"
CONSTANT_STREAK = "challenger_constant_streak"
SHADOW_COLUMNS: list[str] = [
    "week",
    "written_at",
    "window_days",
    "production_hit_rate",
    "challenger_hit_rate",
    DIFF_COLUMN,
    "production_sharpe",
    "challenger_sharpe",
    "n_production",
    "n_challenger",
    "challenger_wins",
    NOT_WIN_STREAK,
    "challenger_prediction_std",
    "challenger_constant",
    CONSTANT_STREAK,
    "gate_skipped_reason",
    "promoted",
    "retry_count",
]

# 影の比べとゲートの成否・今週の challenger の数・失敗の文の頭・勝ちの決め方（#2061）。
# 今の列の並びを変えないよう、右端に足す
STATUS_FLAG_COLUMNS: tuple[str, ...] = (
    "shadow_evaluated",
    "shadow_failed",
    "gate_evaluated",
    "gate_failed",
)
FRESH_COUNT = "fresh_challenger_count"
SHADOW_ERROR = "shadow_error"
SHADOW_ERROR_HEAD = 120
RANK_NUMBER_COLUMNS: tuple[str, ...] = (
    "production_daily_rank_corr",
    "challenger_daily_rank_corr",
)
WIN_RULE = "win_rule"
SHADOW_COLUMNS += [
    *STATUS_FLAG_COLUMNS,
    FRESH_COUNT,
    SHADOW_ERROR,
    *RANK_NUMBER_COLUMNS,
    WIN_RULE,
]


def load_shadow_records(shadow_dir: str) -> list[dict]:
    """shadow_dir の YYYY-Www.json を週の順に読む（読めないものは飛ばす）。"""
    records: list[dict] = []
    for path in sorted(glob.glob(os.path.join(shadow_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning("影の比べの JSON を読めず飛ばす: %s (%s)", path, e)
            continue
        if not isinstance(record, dict):
            logger.warning("影の比べの JSON の形が違うため飛ばす: %s", path)
            continue
        record.setdefault("week", os.path.splitext(os.path.basename(path))[0])
        records.append(record)
    return records


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _flag(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _row(record: dict) -> dict:
    row: dict[str, Any] = {"week": str(record.get("week", ""))}
    for column in NUMBER_COLUMNS:
        row[column] = _number(record.get(column))
    for column in FLAG_COLUMNS:
        row[column] = _flag(record.get(column))
    for column in TEXT_COLUMNS:
        row[column] = _text(record.get(column))
    production = row["production_hit_rate"]
    challenger = row["challenger_hit_rate"]
    row[DIFF_COLUMN] = None if production is None or challenger is None else challenger - production
    for column in STATUS_FLAG_COLUMNS:
        row[column] = _flag(record.get(column))
    fresh = record.get("fresh_challengers")
    row[FRESH_COUNT] = float(len(fresh)) if isinstance(fresh, list) else None
    error = _text(record.get(SHADOW_ERROR))
    row[SHADOW_ERROR] = None if error is None else error[:SHADOW_ERROR_HEAD]
    for column in RANK_NUMBER_COLUMNS:
        row[column] = _number(record.get(column))
    row[WIN_RULE] = _text(record.get(WIN_RULE))
    return row


def _streaks(flags: list[bool | None], target: bool) -> list[int]:
    """flags が target と同じ値の週が、その週まで何週続いたかを返す。"""
    out: list[int] = []
    count = 0
    for flag in flags:
        count = count + 1 if flag is target else 0
        out.append(count)
    return out


def build_shadow_trend(records: list[dict]) -> pd.DataFrame:
    """週ごとの1行の表を週の順に返す（列は SHADOW_COLUMNS）。"""
    rows = sorted((_row(r) for r in records), key=lambda r: r["week"])
    wins = [r["challenger_wins"] for r in rows]
    constants = [r["challenger_constant"] for r in rows]
    for row, not_win, constant in zip(rows, _streaks(wins, False), _streaks(constants, True)):
        row[NOT_WIN_STREAK] = not_win
        row[CONSTANT_STREAK] = constant
    df = pd.DataFrame(rows, columns=SHADOW_COLUMNS)
    for column in [*NUMBER_COLUMNS, DIFF_COLUMN, FRESH_COUNT, *RANK_NUMBER_COLUMNS]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
    for column in (NOT_WIN_STREAK, CONSTANT_STREAK):
        df[column] = df[column].astype(int)
    for column in list(FLAG_COLUMNS) + list(TEXT_COLUMNS):
        df[column] = df[column].astype(object)
    # 文字の列は pandas が空を NaN にするため、足した列は空を None にそろえる
    for column in [*STATUS_FLAG_COLUMNS, SHADOW_ERROR, WIN_RULE]:
        values = df[column].astype(object)
        df[column] = values.where(values.notna(), None)
    return df.reset_index(drop=True)


def load_shadow_trend(shadow_dir: str) -> pd.DataFrame:
    """shadow_dir の JSON を読み、週ごとの表を返す。"""
    return build_shadow_trend(load_shadow_records(shadow_dir))


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.4g}"
    return str(value)


def format_shadow_markdown(df: pd.DataFrame) -> str:
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
