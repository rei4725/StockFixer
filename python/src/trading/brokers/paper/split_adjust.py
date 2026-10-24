"""ペーパーの保有を株式分割で直す（#898）

paper_positions の株数と買値は約定の時の値段のままだが、現在値は分割で調整し直した値段で来る。
分割の記録で保有を直し、記録が取れない間は、買値と現在値の比が分割の比ほど違う保有を評価しない。
"""

import math
from datetime import date
from typing import Any

import pandas as pd

from src.domain.ports import MarketDataPort
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 買値と現在値の比がこれ以上なら、分割による調整の違いを疑う（1:2 の分割の比から少し下）
_SPLIT_SUSPECT_RATIO = 1.8
# 比を分割の比（0.5 刻み）に丸めたときに許す、丸めとの相対の差（採点の側と同じ）
_SPLIT_RATIO_TOLERANCE = 0.1
_JST = "Asia/Tokyo"


def _to_jst_date(value: Any) -> date:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(_JST)
    return ts.date()


def split_factor_since(splits: Any, since: Any) -> float | None:
    """since の日より後に権利落ちした分割の比の積。記録が分からなければ None、無ければ 1.0"""
    if not isinstance(splits, pd.Series):
        return None
    since_date = _to_jst_date(since)
    factor = 1.0
    for ex_date, ratio in splits.items():
        r = float(ratio)
        if r > 0 and not math.isnan(r) and _to_jst_date(ex_date) > since_date:
            factor *= r
    return factor


def looks_like_split_gap(avg_price: float, current_price: float) -> bool:
    """買値と現在値の比が分割の比（0.5 刻み）ほど違うか"""
    if avg_price <= 0 or current_price <= 0 or math.isnan(current_price):
        return False
    gap = avg_price / current_price
    ratio = max(gap, 1 / gap)
    if ratio < _SPLIT_SUSPECT_RATIO:
        return False
    split_ratio = round(ratio * 2) / 2
    return abs(ratio / split_ratio - 1) <= _SPLIT_RATIO_TOLERANCE


def adjust_position_for_splits(
    market_data: MarketDataPort, sym: str, qty: int, avg: float, updated_at: Any
) -> tuple[int, float, bool]:
    """保有の後の分割で株数と買値を直して書き戻す。戻り値の3つ目は分割の記録が取れたか"""
    try:
        splits = market_data.get_splits(f"{sym}.T")
    except Exception:
        logger.warning("[paper] %s: 分割の記録の取得失敗", sym, exc_info=True)
        splits = None
    factor = split_factor_since(splits, updated_at)
    if factor is None:
        return qty, avg, False
    if factor == 1.0:
        return qty, avg, True
    new_qty = int(round(qty * factor))
    new_avg = avg / factor
    with _db_connection() as con:
        con.execute(
            "UPDATE paper_positions SET qty=%s, avg_price=%s, "
            "updated_at=CURRENT_TIMESTAMP WHERE symbol=%s",
            [new_qty, new_avg, sym],
        )
    logger.warning(
        f"[paper] {sym}: 株式分割（比 {factor:g}）で保有を直した "
        f"{qty}株 @ {avg:.1f} → {new_qty}株 @ {new_avg:.1f}"
    )
    return new_qty, new_avg, True
