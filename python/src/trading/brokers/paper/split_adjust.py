"""ペーパーの保有を株式分割で直す（#898）

paper_positions の株数と買値は約定の時の値段のままだが、現在値は分割で調整し直した値段で来る。
分割の記録で保有を直し、記録が取れない間は、買値と現在値の比が分割の比ほど違う保有を評価しない。
"""

import math
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from src.domain.ports import MarketDataPort
from src.trading.brokers.base import OrderSide
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 買値と現在値の比がこれ以上なら、分割による調整の違いを疑う（1:2 の分割の比から少し下）
_SPLIT_SUSPECT_RATIO = 1.8
# 比を分割の比（0.5 刻み）に丸めたときに許す、丸めとの相対の差（採点の側と同じ）
_SPLIT_RATIO_TOLERANCE = 0.1
_JST = "Asia/Tokyo"


def _now_utc() -> datetime:
    """DB の updated_at と同じ、tz 無しの UTC の今の時刻"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


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
    return _adjust_holding(market_data, sym, qty, avg, updated_at, short=False)


def adjust_short_position_for_splits(
    market_data: MarketDataPort, sym: str, qty: int, avg: float, updated_at: Any
) -> tuple[int, float, bool]:
    """空売りの保有を、建てた後の分割で直して書き戻す（#1086）。戻り値は買い持ちと同じ"""
    return _adjust_holding(market_data, sym, qty, avg, updated_at, short=True)


def adjust_stored_short_for_splits(market_data: MarketDataPort, sym: str) -> Any:
    """新しい空売りを仮登録で平均する前に、今の空売りの保有を分割で直す（#1086）

    仮登録で注文の株数を保有の行の単位に直せるよう、分割の記録（取れなければ None）を返す。
    """
    with _db_connection() as con:
        pos = con.execute(
            "SELECT qty, avg_short_price, updated_at FROM paper_short_positions WHERE symbol=%s",
            [sym],
        ).fetchone()
    if pos:
        pos_qty, pos_avg, pos_updated_at = pos
        _adjust_holding(market_data, sym, pos_qty, pos_avg, pos_updated_at, short=True)
    return fetch_splits(market_data, sym)


def split_factor_between(splits: Any, since: Any, until: Any) -> float:
    """since の日より後で until の日までに権利落ちした分割の比の積（#1339）

    約定待ちの注文（created_at の日の単位）を、保有の行（updated_at の日の単位）の株数に直す比。
    記録が分からないとき、until が since より前のときは 1.0。
    """
    after_since = split_factor_since(splits, since)
    after_until = split_factor_since(splits, until)
    if after_since is None or after_until is None or after_since <= after_until:
        return 1.0
    return after_since / after_until


def _adjust_holding(
    market_data: MarketDataPort,
    sym: str,
    qty: int,
    avg: float,
    updated_at: Any,
    *,
    short: bool,
) -> tuple[int, float, bool]:
    splits = fetch_splits(market_data, sym)
    factor = split_factor_since(splits, updated_at)
    if factor is None:
        return qty, avg, False
    if factor == 1.0:
        return qty, avg, True
    if short:
        select_sql = (
            "SELECT qty, avg_short_price, updated_at FROM paper_short_positions "
            "WHERE symbol=%s FOR UPDATE"
        )
        update_sql = (
            "UPDATE paper_short_positions SET qty=%s, avg_short_price=%s, "
            "updated_at=%s WHERE symbol=%s"
        )
    else:
        select_sql = (
            "SELECT qty, avg_price, updated_at FROM paper_positions WHERE symbol=%s FOR UPDATE"
        )
        update_sql = (
            "UPDATE paper_positions SET qty=%s, avg_price=%s, updated_at=%s WHERE symbol=%s"
        )
    # 分割の記録を取る間に約定が行を変えうるので、書く前に行を押さえて読み直し、
    # 読み直した株数と updated_at から比を当て直す（#1423）
    with _db_connection() as con, con.transaction():
        row = con.execute(select_sql, [sym]).fetchone()
        if row is None:
            logger.warning(f"[paper] {sym}: 分割の直しの前に保有の行が無くなったため書かない")
            return int(round(qty * factor)), avg / factor, True
        qty, avg, updated_at = row
        factor = split_factor_since(splits, updated_at) or 1.0
        if factor == 1.0:
            return qty, avg, True
        new_qty = int(round(qty * factor))
        new_avg = avg / factor
        con.execute(update_sql, [new_qty, new_avg, _stamp_after_splits(splits, updated_at), sym])
    label = "空売りの保有" if short else "保有"
    logger.warning(
        f"[paper] {sym}: 株式分割（比 {factor:g}）で{label}を直した "
        f"{qty}株 @ {avg:.1f} → {new_qty}株 @ {new_avg:.1f}"
    )
    return new_qty, new_avg, True


def _stamp_after_splits(splits: pd.Series, since: Any) -> datetime:
    """直した保有に書き戻す updated_at（#1326）

    split_factor_since は tz 無しの値を日付のまま読むので、権利落ちの日の 9:00 JST より前に
    書いた今の時刻は前の日と読まれ、同じ分割をまた当てる。当てた権利落ちの日より前に読まれない値にする。
    """
    since_date = _to_jst_date(since)
    applied = [
        _to_jst_date(ex_date)
        for ex_date, ratio in splits.items()
        if float(ratio) > 0 and _to_jst_date(ex_date) > since_date
    ]
    now = _now_utc()
    if not applied:
        return now
    return max(now, datetime.combine(max(applied), datetime.min.time()))


def fetch_splits(market_data: MarketDataPort, sym: str) -> Any:
    try:
        return market_data.get_splits(f"{sym}.T")
    except Exception:
        logger.warning("[paper] %s: 分割の記録の取得失敗", sym, exc_info=True)
        return None


def adjust_order_for_splits(
    market_data: MarketDataPort, sym: str, order: tuple[Any, ...], open_price: float
) -> tuple[int, float, float | None] | None:
    """約定の前に、保有と注文を株式分割で直す（#1073）。約定させないときは None

    order は (side, qty, 指値, signal_price, created_at)。戻り値は直した (qty, 指値, signal_price)。
    保有は get_positions と同じ直しにかけ、注文は出した後に権利落ちした分割の比で
    株数・指値・signal_price を直す。分割の記録が取れず、値段が分割の比ほど違うときは約定させない。
    SHORT・SHORT_COVER は空売りの保有を同じ直しにかける（#1086）。
    """
    side, qty, limit_price, signal_price, created_at = order
    short = side in (int(OrderSide.SHORT), int(OrderSide.SHORT_COVER))
    if short:
        sql = "SELECT qty, avg_short_price, updated_at FROM paper_short_positions WHERE symbol=%s"
    elif side in (int(OrderSide.BUY), int(OrderSide.SELL)):
        sql = "SELECT qty, avg_price, updated_at FROM paper_positions WHERE symbol=%s"
    else:
        return qty, limit_price, signal_price
    with _db_connection() as con:
        pos = con.execute(sql, [sym]).fetchone()
    pos_avg = None
    if pos:
        pos_qty, raw_avg, pos_updated_at = pos
        _, pos_avg, _ = _adjust_holding(
            market_data, sym, pos_qty, raw_avg, pos_updated_at, short=short
        )
    factor = split_factor_since(fetch_splits(market_data, sym), created_at)
    if factor is None:
        refs = [signal_price, limit_price, pos_avg]
        if any(ref and looks_like_split_gap(float(ref), open_price) for ref in refs):
            logger.warning(
                f"[paper] {sym}: 分割の記録が取れず、注文か保有の値段と始値の比が"
                "分割の比ほど違うため約定スキップ（pending 据え置き）"
            )
            return None
        return qty, limit_price, signal_price
    if factor == 1.0:
        return qty, limit_price, signal_price
    new_qty = int(round(qty * factor))
    new_signal = None if signal_price is None else float(signal_price) / factor
    logger.warning(f"[paper] {sym}: 株式分割（比 {factor:g}）で注文を直した {qty}株 → {new_qty}株")
    return new_qty, float(limit_price or 0.0) / factor, new_signal
