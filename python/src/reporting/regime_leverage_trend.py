"""紙のレジームレバレッジ戦略の記録 regime_leverage_log を読み返し、日ごとの流れの表を作る（#2094）。

書く側（trading/regime_leverage_strategy/service.py）は、週次と日次の判定のたびに1行を足す。
ここでは読むだけで、日本時間の日付ごとに最後の行（id が一番大きい行）の状態を1行にした表を作る。
本番のジョブには組み込まない。

- executed_at は UTC の時刻として読み、日本時間の日付で日を分ける
- 同じ日に複数の行があれば、action・reason・数値は最後の行から取り、その日の理由を reasons に
  並べる（例: "daily_noop; regime_entry"）
- change_from_entry は equity_now_jpy / equity_at_entry_jpy - 1。建玉が無く equity_at_entry_jpy が
  無い（または 0 の）日は空にする

BC の独立の契約により trading の型は import せず、同じ名前の属性を持つ行を受け取る。
"""

import math
from datetime import date, datetime, timezone
from typing import Any, Iterable, Optional, Protocol
from zoneinfo import ZoneInfo

import pandas as pd

JST = ZoneInfo("Asia/Tokyo")
REASON_SEPARATOR = "; "
_COLUMNS = [
    "date",
    "rows",
    "action",
    "reason",
    "reasons",
    "shares",
    "equity_now_jpy",
    "equity_at_entry_jpy",
    "change_from_entry",
    "maintenance_ratio",
    "stop_price_jpy",
]


class RegimeLeverageLogRow(Protocol):
    """regime_leverage_log の1行のうち、表に使う属性（読むだけ）。"""

    @property
    def id(self) -> int: ...

    @property
    def executed_at(self) -> datetime: ...

    @property
    def action(self) -> str: ...

    @property
    def reason(self) -> str: ...

    @property
    def shares(self) -> float: ...

    @property
    def equity_now_jpy(self) -> float: ...

    @property
    def equity_at_entry_jpy(self) -> Optional[float]: ...

    @property
    def stop_price_jpy(self) -> Optional[float]: ...

    @property
    def maintenance_ratio(self) -> Optional[float]: ...


def to_jst_date(executed_at: datetime) -> date:
    """UTC の時刻（naive は UTC とみなす）を日本時間の日付にする。"""
    if executed_at.tzinfo is None:
        executed_at = executed_at.replace(tzinfo=timezone.utc)
    return executed_at.astimezone(JST).date()


def _change_from_entry(equity_now: float, equity_at_entry: Optional[float]) -> Optional[float]:
    if equity_at_entry is None or equity_at_entry == 0:
        return None
    return equity_now / equity_at_entry - 1.0


def build_regime_leverage_trend(rows: Iterable[RegimeLeverageLogRow]) -> pd.DataFrame:
    """日本時間の日付ごとに、最後の行の状態と、その日の行の数・理由の並びを1行にした表を返す。"""
    by_day: dict[date, list[RegimeLeverageLogRow]] = {}
    for row in rows:
        by_day.setdefault(to_jst_date(row.executed_at), []).append(row)
    records = []
    for day in sorted(by_day):
        day_rows = sorted(by_day[day], key=lambda r: r.id)
        last = day_rows[-1]
        records.append(
            {
                "date": day.isoformat(),
                "rows": len(day_rows),
                "action": last.action,
                "reason": last.reason,
                "reasons": REASON_SEPARATOR.join(r.reason for r in day_rows),
                "shares": last.shares,
                "equity_now_jpy": last.equity_now_jpy,
                "equity_at_entry_jpy": last.equity_at_entry_jpy,
                "change_from_entry": _change_from_entry(
                    last.equity_now_jpy, last.equity_at_entry_jpy
                ),
                "maintenance_ratio": last.maintenance_ratio,
                "stop_price_jpy": last.stop_price_jpy,
            }
        )
    df = pd.DataFrame(records, columns=_COLUMNS)
    if not df.empty:
        df["rows"] = df["rows"].astype(int)
    return df


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.4f}"
    return str(value)


def format_regime_leverage_trend_markdown(df: pd.DataFrame) -> str:
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
