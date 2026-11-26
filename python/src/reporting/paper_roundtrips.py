"""紙の注文を FIFO で往復（買いから売り）に組み、入った道別・月別にまとめる（#1817）

読むだけの報告である。発注・約定・paper_orders への書き込みの道には組み込まない。

道は買いの注文の時刻（UTC）で見分ける。8:50 JST の予想の道は 23 時台 UTC、
16:30 JST のルールの道は 7 時台 UTC に注文を作る。ほかの時刻は「不明」にまとめる。

紙の約定（trading/brokers/paper/fill.py）は売りの実現損益を平均の取得値で計算するが、
ここでは FIFO で買いの値段を当てる。銘柄ごとに売り切るまでの損益の合計は一致し、
売り切っていない銘柄では、往復ごとの損益の分け方と合計が平均の取得値とずれる。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import pandas as pd

ROUTE_PREDICTION = "prediction"
ROUTE_RULE = "rule"
ROUTE_UNKNOWN = "unknown"

_PREDICTION_HOUR_UTC = 23
_RULE_HOUR_UTC = 7
_BUY = 1
_SELL = 2

SplitLoader = Callable[[str, str], Optional[pd.Series]]

PAPER_ROUNDTRIP_ORDERS_SQL = """
    SELECT order_id, market, symbol, side, qty, COALESCE(fill_price, price) AS fill_price,
           created_at, filled_at, realized_pnl
    FROM paper_orders
    WHERE status = 'filled' AND side IN (%s, %s) AND qty > 0
    ORDER BY COALESCE(filled_at, created_at), created_at, order_id
"""


@dataclass(frozen=True)
class OpenLot:
    """まだ売りと対になっていない買いの残り"""

    symbol: str
    qty: int
    price: float
    route: str
    entered_at: datetime


@dataclass(frozen=True)
class RoundTrip:
    """1つの買いの一部または全部と、1つの売りの組み"""

    symbol: str
    qty: int
    route: str
    entry_price: float
    exit_price: float
    entered_at: datetime
    exited_at: datetime

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.qty

    @property
    def holding_days(self) -> int:
        return (self.exited_at.date() - self.entered_at.date()).days

    @property
    def exit_month(self) -> str:
        return self.exited_at.strftime("%Y-%m")

    @property
    def return_rate(self) -> Optional[float]:
        """出口 / 入口 - 1。入口が 0 以下なら None"""
        if self.entry_price <= 0:
            return None
        return self.exit_price / self.entry_price - 1


@dataclass(frozen=True)
class RoundTripGroup:
    """往復のまとめの1行（道別なら key は道、月別なら key は月）

    gross_profit は勝ち（損益 > 0）の損益の合計、gross_loss は負け（損益 < 0）の
    損益の合計の絶対値。損益が 0 の往復はどちらにも入れない（#2179）。
    """

    route: str
    key: str
    trips: int
    wins: int
    pnl: float
    avg_holding_days: float
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    avg_win_return: Optional[float] = None
    avg_loss_return: Optional[float] = None

    @property
    def win_rate(self) -> float:
        return self.wins / self.trips if self.trips else 0.0

    @property
    def profit_factor(self) -> Optional[float]:
        """総利益 / 総損失。総損失が 0 なら None"""
        return self.gross_profit / self.gross_loss if self.gross_loss > 0 else None


@dataclass(frozen=True)
class RoundTripSummary:
    """道別と、道と月の組みのまとめ"""

    by_route: list[RoundTripGroup] = field(default_factory=list)
    by_month: list[RoundTripGroup] = field(default_factory=list)
    total_pnl: float = 0.0


def classify_route(created_at: datetime) -> str:
    """注文の時刻（UTC。時差の無い値は UTC とみなす）から入った道を返す"""
    ts = created_at
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc)
    if ts.hour == _PREDICTION_HOUR_UTC:
        return ROUTE_PREDICTION
    if ts.hour == _RULE_HOUR_UTC:
        return ROUTE_RULE
    return ROUTE_UNKNOWN


def _ts(value: Any) -> datetime:
    return pd.Timestamp(value).to_pydatetime()


def _is_phantom_sell(row: Any) -> bool:
    """保有が無く株数を動かさなかった売りか（fill.py は realized_pnl を書かずに filled にする）"""
    return hasattr(row, "realized_pnl") and pd.isna(row.realized_pnl)


def count_phantom_sells(orders: pd.DataFrame) -> int:
    """realized_pnl の無い約定した売り（株数を動かさなかった売り）の件数"""
    if orders.empty or "realized_pnl" not in orders.columns:
        return 0
    sells = orders[orders["side"].astype(int) == _SELL]
    return int(sells["realized_pnl"].isna().sum())


def _split_factors(splits: Optional[pd.Series]) -> Callable[[pd.Timestamp], float]:
    """約定の時刻 -> その時刻より後の分割の比の積、を返す関数を作る

    trading/paper_equity._split_factors と同じ決まり（BC をまたいで import しないため
    同じ形でここに置く）。分割より前の約定の株数を積の倍に、値段を積で割ると今の水準になる。
    """
    if splits is None or splits.empty:
        return lambda day: 1.0
    ratios = pd.to_numeric(splits, errors="coerce")
    ratios = ratios[ratios > 0]

    def factor(day: pd.Timestamp) -> float:
        return float(ratios[ratios.index > day].prod())

    return factor


def _naive_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def build_roundtrips(
    orders: pd.DataFrame,
    split_loader: Optional[SplitLoader] = None,
) -> tuple[list[RoundTrip], list[OpenLot], dict[str, int]]:
    """約定した買いと売りを銘柄ごとに FIFO で組む

    戻り値は（往復、開いたままの買い、買いの残りを超えた売りの株数の銘柄別）。
    realized_pnl の列があり、その値が空の売りは株数を動かさなかった売りとして飛ばす。
    split_loader（(market, symbol) -> 分割の比の Series か None）と market の列があれば、
    約定の株数を約定より後の分割の比の積の倍に、値段を積で割って今の水準に直してから組む
    （紙のブローカーは分割で保有を直し、分割後の売りはその株数で約定するため）。
    """
    trips: list[RoundTrip] = []
    unmatched: dict[str, int] = {}
    lots: dict[str, deque[OpenLot]] = {}
    if orders.empty:
        return trips, [], unmatched
    df = orders.copy()
    df["when_at"] = df["filled_at"].where(df["filled_at"].notna(), df["created_at"])
    df = df.sort_values(["when_at", "created_at", "order_id"], kind="stable")
    use_splits = split_loader is not None and "market" in df.columns
    factors: dict[tuple[str, str], Callable[[pd.Timestamp], float]] = {}
    for row in df.itertuples(index=False):
        symbol = str(row.symbol)
        qty = int(row.qty)
        price = float(row.fill_price)
        when = _ts(row.when_at)
        if use_splits and split_loader is not None:
            key = (str(row.market), symbol)
            if key not in factors:
                factors[key] = _split_factors(split_loader(*key))
            factor = factors[key](_naive_utc(row.when_at))
            if factor != 1.0:
                qty = int(round(qty * factor))
                price = price / factor
        queue = lots.setdefault(symbol, deque())
        if int(row.side) == _BUY:
            route = classify_route(_ts(row.created_at))
            queue.append(OpenLot(symbol, qty, price, route, when))
            continue
        if int(row.side) != _SELL or _is_phantom_sell(row):
            continue
        remaining = qty
        while remaining > 0 and queue:
            lot = queue[0]
            take = min(remaining, lot.qty)
            trips.append(RoundTrip(symbol, take, lot.route, lot.price, price, lot.entered_at, when))
            remaining -= take
            if take == lot.qty:
                queue.popleft()
            else:
                queue[0] = OpenLot(symbol, lot.qty - take, lot.price, lot.route, lot.entered_at)
        if remaining > 0:
            unmatched[symbol] = unmatched.get(symbol, 0) + remaining
    open_lots = [lot for queue in lots.values() for lot in queue]
    return trips, open_lots, unmatched


def _mean_return(trips: list[RoundTrip]) -> Optional[float]:
    """往復ごとの騰落率の単純平均。入口が 0 以下の往復は入れない。無ければ None"""
    rates = [r for r in (t.return_rate for t in trips) if r is not None]
    return sum(rates) / len(rates) if rates else None


def _group(route: str, key: str, trips: list[RoundTrip]) -> RoundTripGroup:
    n = len(trips)
    winners = [t for t in trips if t.pnl > 0]
    losers = [t for t in trips if t.pnl < 0]
    return RoundTripGroup(
        route=route,
        key=key,
        trips=n,
        wins=sum(1 for t in trips if t.pnl > 0),
        pnl=float(sum(t.pnl for t in trips)),
        avg_holding_days=(sum(t.holding_days for t in trips) / n) if n else 0.0,
        gross_profit=float(sum(t.pnl for t in winners)),
        gross_loss=float(abs(sum(t.pnl for t in losers))),
        avg_win_return=_mean_return(winners),
        avg_loss_return=_mean_return(losers),
    )


def summarize_roundtrips(trips: list[RoundTrip]) -> RoundTripSummary:
    """往復を道別と、道と月の組みでまとめる"""
    routes: dict[str, list[RoundTrip]] = {}
    months: dict[tuple[str, str], list[RoundTrip]] = {}
    for t in trips:
        routes.setdefault(t.route, []).append(t)
        months.setdefault((t.route, t.exit_month), []).append(t)
    return RoundTripSummary(
        by_route=[_group(r, r, ts) for r, ts in sorted(routes.items())],
        by_month=[_group(r, m, ts) for (r, m), ts in sorted(months.items())],
        total_pnl=float(sum(t.pnl for t in trips)),
    )


def load_paper_roundtrip_orders(con: Any) -> pd.DataFrame:
    """paper_orders から約定した買いと売りを読む（読むだけ）"""
    cur = con.execute(PAPER_ROUNDTRIP_ORDERS_SQL, [_BUY, _SELL])
    columns = [d[0] for d in cur.description]
    return pd.DataFrame(cur.fetchall(), columns=columns)
