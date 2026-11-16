"""紙の口座の入金の出来事（paper_cash_events）の読み出しと、約定からの現金のたどり直し（#1877）。

元手 100 万円の口座に、移行 0010 が 900 万円を足し、その日と額を paper_cash_events に残す。
評価額の流れと週の成績は、入金をその日の現金に足し、利益には数えない。
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import pandas as pd

from src.trading.brokers.base import OrderSide
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

Deposit = tuple[pd.Timestamp, float]
SplitLoader = Callable[[str, str], Optional[pd.Series]]


def _split_factors(splits: Optional[pd.Series]) -> Callable[[pd.Timestamp], float]:
    """約定の日 -> その日より後の分割の比の積、を返す関数を作る。

    raw の終値は分割があると全期間を調整し直してあるため（#817）、分割より前の約定の
    株数をこの積の倍に、値段を積で割ると、終値と同じ水準になる（#899）。
    評価の最後の日より後の分割も終値に入っているため数える（前の週の計算し直し。#1695）。
    評価額の流れ（paper_equity）と約定のたどり直し（replay_paper_cash）が同じ道で使う（#1912）。
    """
    if splits is None or splits.empty:
        return lambda day: 1.0
    ratios = pd.to_numeric(splits, errors="coerce")
    ratios = ratios[ratios > 0]

    def factor(day: pd.Timestamp) -> float:
        return float(ratios[ratios.index > day].prod())

    return factor


def load_cash_events() -> list[Deposit]:
    """入金の出来事を時刻の順に返す。表が無いか読めなければ空（評価の道は止めない）。"""
    try:
        with _db_connection() as con:
            df = pd.read_sql(
                "SELECT occurred_at, amount FROM paper_cash_events ORDER BY occurred_at", con
            )
    except Exception:
        logger.warning("紙の口座の入金の出来事が読めないため入金なしで評価", exc_info=True)
        return []
    return [(_naive_day(row["occurred_at"]), float(row["amount"])) for _, row in df.iterrows()]


def _naive_day(value: object) -> pd.Timestamp:
    """約定の filled_at と同じく、時刻帯を外して日に丸める。"""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


def deposits_between(
    deposits: Sequence[Deposit], after: Optional[pd.Timestamp], until: pd.Timestamp
) -> float:
    """after より後（None なら最初から）until までの入金の合計を返す。"""
    return float(
        sum(
            amount
            for day, amount in deposits
            if (after is None or pd.Timestamp(day) > after) and pd.Timestamp(day) <= until
        )
    )


def first_deposit_day(deposits: Sequence[Deposit]) -> Optional[pd.Timestamp]:
    """最初の入金の日（切り替えの日）を返す。入金が無ければ None。"""
    if not deposits:
        return None
    return min(pd.Timestamp(day).normalize() for day, _ in deposits)


def add_deposits_to_cash(
    cash_events: dict[pd.Timestamp, float],
    deposits: Sequence[Deposit],
    index: pd.DatetimeIndex,
) -> None:
    """入金を、約定と同じく直近の評価日に丸めて日次の現金の増減に足す。"""
    for raw_day, amount in deposits:
        day = pd.Timestamp(raw_day).normalize()
        if day > index[-1]:
            continue
        pos = int(index.searchsorted(max(day, index[0])))
        if pos >= len(index):
            continue
        key = index[pos]
        cash_events[key] = cash_events.get(key, 0.0) + float(amount)


def _factor_for(
    factors: dict[str, Callable[[pd.Timestamp], float]],
    key: str,
    split_loader: Optional[SplitLoader],
) -> Callable[[pd.Timestamp], float]:
    """「市場/銘柄」の分割の比の関数を、銘柄ごとに1度だけ取り出して返す。"""
    if key not in factors:
        market, _, symbol = key.partition("/")
        splits = split_loader(market, symbol) if split_loader is not None else None
        factors[key] = _split_factors(splits)
    return factors[key]


def replay_paper_cash(
    orders: pd.DataFrame,
    initial_balance: float,
    deposits: Sequence[Deposit] = (),
    split_loader: Optional[SplitLoader] = None,
) -> float:
    """約定（と入金）を時刻の順に元手からたどり、紙の口座の現金を返す（純粋ロジック）。

    brokers/paper/fill.py と同じ決まり: 手数料なし。BUY は −株数×値段、SELL は持ち分までの
    株数×値段、SHORT は現金を動かさず、SHORT_COVER は（建値 − 値段）× 返済株数を足す。
    買いの持ち分は、paper_broker・評価額の流れと同じく分割の比で直した株数で数える
    （#899・#1899・#1912）。split_loader が None か、記録が取れない銘柄は生の株数で数える。
    """
    # (時刻, 種類 0=入金 1=約定, 値段か入金額, 売買, 株数, 銘柄)
    events: list[tuple[pd.Timestamp, int, float, int, int, str]] = []
    for _, row in orders.iterrows():
        when = pd.Timestamp(row["filled_at"])
        if when.tzinfo is not None:
            when = when.tz_localize(None)
        key = f"{row['market']}/{row['symbol']}"
        events.append((when, 1, float(row["fill_price"]), int(row["side"]), int(row["qty"]), key))
    for day, amount in deposits:
        events.append((pd.Timestamp(day), 0, float(amount), -1, 0, ""))
    events.sort(key=lambda e: (e[0], e[1]))

    cash = float(initial_balance)
    long_qty: dict[str, float] = {}
    short_qty: dict[str, int] = {}
    short_cost: dict[str, float] = {}
    factors: dict[str, Callable[[pd.Timestamp], float]] = {}
    for when, kind, price, side, qty, key in events:
        if kind == 0:
            cash += price
        elif side == int(OrderSide.BUY):
            cash -= qty * price
            factor = _factor_for(factors, key, split_loader)(when.normalize())
            long_qty[key] = long_qty.get(key, 0.0) + qty * factor
        elif side == int(OrderSide.SELL):
            factor = _factor_for(factors, key, split_loader)(when.normalize())
            held = long_qty.get(key, 0.0)
            sell = min(float(qty), round(held / factor, 6))
            cash += sell * price
            long_qty[key] = round(held - sell * factor, 6)
        elif side == int(OrderSide.SHORT):
            short_cost[key] = short_cost.get(key, 0.0) + qty * price
            short_qty[key] = short_qty.get(key, 0) + qty
        elif side == int(OrderSide.SHORT_COVER):
            held = short_qty.get(key, 0)
            cover = min(qty, held)
            if cover:
                avg = short_cost[key] / held
                cash += (avg - price) * cover
                short_cost[key] -= avg * cover
                short_qty[key] = held - cover
    return cash
