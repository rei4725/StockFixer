"""
ペーパートレードの約定履歴から日次エクイティ（評価額）系列を再構成する。

paper_orders（status='filled'）を時系列に適用して現金残高と保有数量を復元し、
保有銘柄を日次終値でマークトゥマーケットして評価額を算出する。

制約:
    - ロング（BUY/SELL）のみ対応。SHORT/SHORT_COVER は現金影響のみ反映し、
      空売り建玉の評価損益は含めない（ENABLE_SHORT_SIDE 既定 false のため）。
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from typing import Any, Callable, Optional

import pandas as pd

from config.settings import PAPER_INITIAL_BALANCE
from src.trading.brokers.base import OrderSide
from src.utils.db import load_raw_ohlcv
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 現金増減の符号: BUY/SHORT_COVER は支出、SELL は収入。
# SHORT は売建の受け取り（簡易モデル: 受領現金のみ反映）。
_CASH_SIGN = {
    int(OrderSide.BUY): -1,
    int(OrderSide.SELL): +1,
    int(OrderSide.SHORT): +1,
    int(OrderSide.SHORT_COVER): -1,
}
# 保有数量の増減（ロングのみ）
_QTY_SIGN = {int(OrderSide.BUY): +1, int(OrderSide.SELL): -1}
# 週の成績で市場ごとに比べるベンチマーク（#812）
WEEKLY_BENCHMARKS = {"jp": "1306.T", "us": "^GSPC"}

BenchmarkFetcher = Callable[[str, str, str], dict[str, Any]]


def _load_filled_orders() -> pd.DataFrame:
    """約定済み注文を時系列順に取得する。"""
    with _db_connection() as con:
        df = pd.read_sql(
            """
            SELECT market, symbol, side, qty, fill_price, filled_at
            FROM paper_orders
            WHERE status = 'filled' AND fill_price IS NOT NULL AND filled_at IS NOT NULL
              AND fill_price <> 'NaN'
            ORDER BY filled_at
            """,
            con,
        )
    return df


def _default_price_loader(market: str, symbol: str) -> Optional[pd.Series]:
    """銘柄の日次終値 Series を返す（index=DatetimeIndex）。"""
    df = load_raw_ohlcv(market, symbol)
    if df is None or df.empty or "Close" not in df.columns:
        return None
    series = df["Close"].copy()
    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
    return series


def build_equity_series(
    orders: pd.DataFrame,
    price_loader: Callable[[str, str], Optional[pd.Series]],
    initial_balance: float,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """約定履歴と価格ローダーから日次エクイティ系列を構成する（純粋ロジック）。

    Args:
        orders: market/symbol/side/qty/fill_price/filled_at 列を持つ約定 DataFrame
        price_loader: (market, symbol) -> 日次終値 Series（取得不能時 None）
        initial_balance: 初期資金
        index: 評価日付の DatetimeIndex（昇順・正規化済み）

    Returns:
        index に沿った評価額 Series
    """
    if index.empty:
        return pd.Series(dtype=float)

    work = orders.copy()
    work["filled_at"] = pd.to_datetime(work["filled_at"]).dt.tz_localize(None).dt.normalize()

    # 日次の現金増減と銘柄別数量増減を dict に集計してから Series 化する
    cash_events: dict[pd.Timestamp, float] = {}
    qty_events: dict[tuple[str, str], dict[pd.Timestamp, float]] = {}

    for _, row in work.iterrows():
        side = int(row["side"])
        day = pd.Timestamp(row["filled_at"])
        if day > index[-1]:
            continue
        # index 開始前の約定は初日に寄せて累積に反映する
        if day < index[0]:
            day = index[0]
        # 直近の評価日（営業日）に丸める
        pos = int(index.searchsorted(day))
        if pos >= len(index):
            continue
        day = index[pos]

        amount = float(row["fill_price"]) * int(row["qty"])
        cash_events[day] = cash_events.get(day, 0.0) + _CASH_SIGN.get(side, 0) * amount

        if side in _QTY_SIGN:
            key = (str(row["market"]), str(row["symbol"]))
            bucket = qty_events.setdefault(key, {})
            bucket[day] = bucket.get(day, 0.0) + _QTY_SIGN[side] * int(row["qty"])

    cash_delta = pd.Series(cash_events, dtype=float).reindex(index, fill_value=0.0)
    cash = initial_balance + cash_delta.cumsum()
    equity = cash.copy()

    for (market, symbol), bucket in qty_events.items():
        holdings = pd.Series(bucket, dtype=float).reindex(index, fill_value=0.0).cumsum()
        if (holdings == 0).all():
            continue
        prices = price_loader(market, symbol)
        if prices is None or prices.empty:
            logger.warning("価格データなしのため評価から除外: %s/%s", market, symbol)
            continue
        aligned = prices.reindex(index).ffill().bfill()
        equity += holdings * aligned

    return equity


def get_paper_equity_curve(days: int = 180) -> pd.Series:
    """ペーパートレードの日次エクイティ系列を返す。

    Args:
        days: 評価対象の直近日数

    Returns:
        日次評価額 Series（約定が1件もない場合は空 Series）
    """
    orders = _load_filled_orders()
    if orders.empty:
        logger.info("約定済みペーパー注文がないためエクイティ系列は空")
        return pd.Series(dtype=float)

    end = pd.Timestamp(datetime.now().date())
    first_fill = pd.to_datetime(orders["filled_at"]).min().normalize()
    start = max(first_fill, end - timedelta(days=days))
    index = pd.bdate_range(start=start, end=end)

    return build_equity_series(
        orders,
        _default_price_loader,
        float(PAPER_INITIAL_BALANCE),
        index,
    )


def _fetch_benchmark_return(
    fetcher: BenchmarkFetcher, ticker: str, start: str, end: str
) -> Optional[float]:
    """ベンチマークの期間リターンを返す。取得できなければ None（週の道は止めない）。"""
    try:
        value = fetcher(ticker, start, end).get("total_return")
    except Exception:
        logger.warning("ベンチマーク取得失敗のため null にする: %s", ticker, exc_info=True)
        return None
    return None if value is None else float(value)


def compute_weekly_results(
    orders: pd.DataFrame,
    price_loader: Callable[[str, str], Optional[pd.Series]],
    initial_balance: float,
    as_of: date,
    benchmark_fetcher: BenchmarkFetcher,
) -> dict[str, Any]:
    """as_of 以前の直近の営業日を含む ISO 週の、市場ごとの紙の売買の成績を返す（純粋ロジック）。

    週の騰落率は前週の最終営業日の評価額から今週の金曜の評価額まで。
    ベンチマークも同じ期間（前週の最終営業日の終値 → 今週の終値）で比べる。

    Returns:
        {"week", "start", "end", "markets": {market: {weekly_return, benchmark_ticker,
        benchmark_return, excess_return, fills}}}。取れない値は None
    """
    last_bday = pd.Timestamp(as_of)
    while last_bday.weekday() >= 5:
        last_bday -= pd.Timedelta(days=1)
    week_start = last_bday - pd.Timedelta(days=last_bday.weekday())
    week_end = week_start + pd.Timedelta(days=4)
    base_day = week_start - pd.offsets.BDay(1)
    iso = week_end.isocalendar()

    work = orders.copy()
    if not work.empty:
        work["filled_at"] = pd.to_datetime(work["filled_at"]).dt.tz_localize(None).dt.normalize()

    markets: dict[str, dict[str, Any]] = {}
    for market, ticker in WEEKLY_BENCHMARKS.items():
        market_orders = work[work["market"] == market] if not work.empty else work
        fills = 0
        weekly_return: Optional[float] = 0.0
        if not market_orders.empty:
            in_week = market_orders["filled_at"].between(week_start, week_end)
            fills = int(in_week.sum())
            first_fill = market_orders["filled_at"].min()
            index = pd.bdate_range(start=min(first_fill, base_day), end=week_end)
            equity = build_equity_series(market_orders, price_loader, initial_balance, index)
            base_value = float(equity.loc[base_day])
            end_value = float(equity.loc[week_end])
            weekly_return = end_value / base_value - 1 if base_value else None

        benchmark_return = _fetch_benchmark_return(
            benchmark_fetcher,
            ticker,
            base_day.date().isoformat(),
            (week_end + pd.Timedelta(days=1)).date().isoformat(),
        )
        excess_return = (
            None
            if weekly_return is None or benchmark_return is None
            else weekly_return - benchmark_return
        )
        markets[market] = {
            "weekly_return": None if weekly_return is None else round(weekly_return, 6),
            "benchmark_ticker": ticker,
            "benchmark_return": benchmark_return,
            "excess_return": None if excess_return is None else round(excess_return, 6),
            "fills": fills,
        }

    return {
        "week": f"{iso[0]}-W{iso[1]:02d}",
        "start": week_start.date().isoformat(),
        "end": week_end.date().isoformat(),
        "markets": markets,
    }


def write_paper_weekly_results(
    as_of: date,
    benchmark_fetcher: BenchmarkFetcher,
    output_dir: str,
) -> Optional[str]:
    """紙の売買の週の成績を output_dir/YYYY-Www.json に書き、そのパスを返す。

    約定が1件もない場合は書かずに None を返す。
    """
    orders = _load_filled_orders()
    if orders.empty:
        logger.info("約定済みペーパー注文がないため週の成績は書かない")
        return None

    result = compute_weekly_results(
        orders,
        _default_price_loader,
        float(PAPER_INITIAL_BALANCE),
        as_of,
        benchmark_fetcher,
    )
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{result['week']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("紙の売買の週の成績を書いた: %s", path)
    return path
