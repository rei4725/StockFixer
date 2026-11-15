"""results/orders と results/orders/rule の日ごとの発注の記録を読み返し、流れの表を作る（#1836）。

書く側（trading/execution/gate_record.py）の JSON は1日1ファイルで、新規を飛ばした理由や
保有の判定漏れの流れを見るにはファイルを1つずつ開くしかなかった。ここでは読むだけで、
日・道（8:50 か 16:30）ごとに1行の表を作る。本番のジョブには組み込まない。

- 道は 8:50 の記録（results/orders/*.json）を ROUTE_MORNING、16:30 のルールの道
  （results/orders/rule/*.json）を ROUTE_RULE とする。片方の道しか無い日は、その道の行だけを作る
- mode・market は JSON の値をそのまま写す
- 値が null の数・鍵の無い数は欠け（pd.NA）にし、0 と分ける。skip_reasons の中で鍵の無い理由は
  その日その理由で飛ばした銘柄が無かったので 0、skip_reasons そのものが無い記録は欠けにする
- entry_error・token_error・new_buy_step_error は有無（True/False）に直す。鍵の無い記録は欠け
- 読めないファイル・形の違う JSON は警告を出して飛ばす
"""

import glob
import json
import os
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

ROUTE_MORNING = "0850"
ROUTE_RULE = "1630"

COUNT_COLUMNS: tuple[str, ...] = (
    "buy_orders",
    "sell_orders",
    "short_orders",
    "skipped",
    "errors",
)
FLAG_COLUMNS: tuple[str, ...] = (
    "trading_stopped",
    "correlation_blocked",
    "allow_new_buys",
)
ERROR_COLUMNS: tuple[str, ...] = (
    "entry_error",
    "token_error",
    "new_buy_step_error",
)
HELD_COLUMNS: tuple[str, ...] = (
    "held_without_prediction_count",
    "sl_tp_unevaluable_count",
    "unjudged_held_count",
    "unavailable_held_count",
)
SKIP_PREFIX = "skip_"
_KEY_COLUMNS = ["date", "route"]
_BASE_COLUMNS = (
    _KEY_COLUMNS
    + ["market", "mode"]
    + list(COUNT_COLUMNS)
    + list(FLAG_COLUMNS)
    + list(ERROR_COLUMNS)
    + list(HELD_COLUMNS)
)


def _load_dir(dir_path: str, route: str) -> list[dict]:
    records: list[dict] = []
    for path in sorted(glob.glob(os.path.join(dir_path, "*.json"))):
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
        record["route"] = route
        records.append(record)
    return records


def load_order_gate_records(orders_dir: str) -> list[dict]:
    """orders_dir の 8:50 の記録と orders_dir/rule の 16:30 の記録を読む（読めないものは飛ばす）。

    返す記録には道の印 route（ROUTE_MORNING か ROUTE_RULE）を足す。
    """
    return _load_dir(orders_dir, ROUTE_MORNING) + _load_dir(
        os.path.join(orders_dir, "rule"), ROUTE_RULE
    )


def _count(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return pd.NA
    return int(value)


def _flag(record: dict, key: str) -> Any:
    value = record.get(key)
    return value if isinstance(value, bool) else pd.NA


def _has_error(record: dict, key: str) -> Any:
    if key not in record:
        return pd.NA
    return record[key] is not None


def _row(record: dict) -> dict:
    row: dict[str, Any] = {
        "date": str(record.get("date", "")),
        "route": record["route"],
        "market": record.get("market"),
        "mode": record.get("mode"),
    }
    for key in COUNT_COLUMNS + HELD_COLUMNS:
        row[key] = _count(record.get(key))
    for key in FLAG_COLUMNS:
        row[key] = _flag(record, key)
    for key in ERROR_COLUMNS:
        row[key] = _has_error(record, key)
    return row


def _skip_reasons(record: dict) -> dict[str, int] | None:
    reasons = record.get("skip_reasons")
    if not isinstance(reasons, dict):
        return None
    out: dict[str, int] = {}
    for key, value in reasons.items():
        number = _count(value)
        if number is not pd.NA:
            out[str(key)] = number
    return out


def build_order_gate_trend(records: list[dict]) -> pd.DataFrame:
    """日・道ごとに1行の表を返す。skip_reasons は理由ごとに skip_<理由> の列にする。"""
    rows: list[dict] = []
    reasons_by_row: list[dict[str, int] | None] = []
    for record in records:
        rows.append(_row(record))
        reasons_by_row.append(_skip_reasons(record))
    reason_keys = sorted({k for reasons in reasons_by_row if reasons for k in reasons})
    for row, reasons in zip(rows, reasons_by_row):
        for key in reason_keys:
            row[SKIP_PREFIX + key] = pd.NA if reasons is None else reasons.get(key, 0)
    columns = _BASE_COLUMNS + [SKIP_PREFIX + k for k in reason_keys]
    df = pd.DataFrame(rows, columns=columns)
    for column in list(COUNT_COLUMNS + HELD_COLUMNS) + [SKIP_PREFIX + k for k in reason_keys]:
        df[column] = df[column].astype("Int64")
    for column in FLAG_COLUMNS + ERROR_COLUMNS:
        df[column] = df[column].astype("boolean")
    return df.sort_values(_KEY_COLUMNS, kind="stable").reset_index(drop=True)


def load_order_gate_trend(orders_dir: str) -> pd.DataFrame:
    """orders_dir の JSON を読み、日・道ごとの表を返す。"""
    return build_order_gate_trend(load_order_gate_records(orders_dir))


def _cell(value: Any) -> str:
    if value is None or value is pd.NA:
        return "null"
    return str(value)


def format_order_gate_markdown(df: pd.DataFrame) -> str:
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


def format_order_gate_csv(df: pd.DataFrame) -> str:
    """表を CSV の文字列にする（欠けは null と書き、0 と分ける）。"""
    return df.to_csv(index=False, na_rep="null")
