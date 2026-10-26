"""
ペーパートレードの約定履歴から日次エクイティ（評価額）系列を再構成する。

paper_orders（status='filled'）を時系列に適用して現金残高と保有数量を復元し、
保有銘柄を日次終値でマークトゥマーケットして評価額を算出する。
終値の分からない保有日は、最後に分かる終値、それも無ければ約定の値段で評価する。

空売り（SHORT/SHORT_COVER）の建玉は負の株数として同じく時価で評価し、
売り代金を足した現金から返済の義務（株数 × 終値）を引く（#1163）。
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from typing import Any, Callable, Optional

import pandas as pd
import yfinance as yf

from config.settings import PAPER_INITIAL_BALANCE
from src.trading.brokers.base import OrderSide
from src.utils.data_path_utils import get_ticker
from src.utils.db import load_raw_ohlcv
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 現金増減の符号: BUY/SHORT_COVER は支出、SELL は収入。
# SHORT は売建の受け取り。
_CASH_SIGN = {
    int(OrderSide.BUY): -1,
    int(OrderSide.SELL): +1,
    int(OrderSide.SHORT): +1,
    int(OrderSide.SHORT_COVER): -1,
}
# 保有数量の増減。空売りの建玉は負の株数として時価で評価する（#1163）
_QTY_SIGN = {
    int(OrderSide.BUY): +1,
    int(OrderSide.SELL): -1,
    int(OrderSide.SHORT): -1,
    int(OrderSide.SHORT_COVER): +1,
}
# 週の成績で市場ごとに比べるベンチマーク（#812）
WEEKLY_BENCHMARKS = {"jp": "1306.T", "us": "^GSPC"}
# ベンチマークの起点の終値を base_day からさかのぼって探す営業日の数（年末年始や大型連休の休場をまたぐ幅）
_BENCHMARK_LOOKBACK_BDAYS = 10

BenchmarkFetcher = Callable[[str, str, str], dict[str, Any]]
SplitLoader = Callable[[str, str], Optional[pd.Series]]


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


def _default_split_loader(market: str, symbol: str) -> Optional[pd.Series]:
    """銘柄の分割の記録 Series を yfinance から返す（index=分割の日、値=分割の比）。

    取れなければ None（分割で直さずに評価する）。
    """
    ticker = get_ticker(market, symbol)
    try:
        splits = yf.Ticker(ticker).splits
    except Exception:
        logger.warning(
            "分割の記録の取得失敗のため直さずに評価: %s/%s", market, symbol, exc_info=True
        )
        return None
    if splits is None or splits.empty:
        return None
    splits = splits.copy()
    idx = pd.to_datetime(splits.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    splits.index = idx.normalize()
    return splits


def _split_factors(
    splits: Optional[pd.Series], end: pd.Timestamp
) -> Callable[[pd.Timestamp], float]:
    """約定の日 -> その日より後で end までの分割の比の積、を返す関数を作る。

    raw の終値は分割があると全期間を調整し直してあるため（#817）、分割より前の約定の
    株数をこの積の倍に、値段を積で割ると、終値と同じ水準になる（#899）。
    """
    if splits is None or splits.empty:
        return lambda day: 1.0
    ratios = pd.to_numeric(splits, errors="coerce")
    ratios = ratios[(ratios > 0) & (ratios.index <= end)]

    def factor(day: pd.Timestamp) -> float:
        return float(ratios[ratios.index > day].prod())

    return factor


def build_equity_series(
    orders: pd.DataFrame,
    price_loader: Callable[[str, str], Optional[pd.Series]],
    initial_balance: float,
    index: pd.DatetimeIndex,
    split_loader: Optional[SplitLoader] = None,
) -> pd.Series:
    """約定履歴と価格ローダーから日次エクイティ系列を構成する（純粋ロジック）。

    Args:
        orders: market/symbol/side/qty/fill_price/filled_at 列を持つ約定 DataFrame
        price_loader: (market, symbol) -> 日次終値 Series（取得不能時 None）
        initial_balance: 初期資金
        index: 評価日付の DatetimeIndex（昇順・正規化済み）
        split_loader: (market, symbol) -> 分割の比の Series（無ければ None）。None なら直さない

    Returns:
        index に沿った評価額 Series
    """
    equity, _, _ = _build_equity_with_fallbacks(
        orders, price_loader, initial_balance, index, split_loader
    )
    return equity


def _build_equity_with_fallbacks(
    orders: pd.DataFrame,
    price_loader: Callable[[str, str], Optional[pd.Series]],
    initial_balance: float,
    index: pd.DatetimeIndex,
    split_loader: Optional[SplitLoader] = None,
) -> tuple[pd.Series, dict[tuple[str, str], pd.DatetimeIndex], list[tuple[str, str]]]:
    """評価額 Series と、約定の値段で評価した (market, symbol) ごとの日付と、
    最後の日に保有しているのにその日以降の終値が無い (market, symbol) を返す。

    保有日の値段は、その日までの最後の終値（過去の方向にだけ埋めた値）で評価する。
    終値が1つも分からない保有日は、0円にせず、その日までの最後の約定の値段で評価する（#850）。
    分割より前の約定の株数と値段は、分割の比で終値と同じ水準に直して評価する（#899）。
    現金は株数×値段で変わらないため直さない。
    """
    if index.empty:
        return pd.Series(dtype=float), {}, []

    work = orders.copy()
    work["filled_at"] = pd.to_datetime(work["filled_at"]).dt.tz_localize(None).dt.normalize()

    # 日次の現金増減と銘柄別数量増減を dict に集計してから Series 化する
    cash_events: dict[pd.Timestamp, float] = {}
    qty_events: dict[tuple[str, str], dict[pd.Timestamp, float]] = {}
    fill_price_events: dict[tuple[str, str], dict[pd.Timestamp, float]] = {}
    split_factors: dict[tuple[str, str], Callable[[pd.Timestamp], float]] = {}

    for _, row in work.iterrows():
        side = int(row["side"])
        day = pd.Timestamp(row["filled_at"])
        fill_day = day
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
            if key not in split_factors:
                splits = split_loader(*key) if split_loader is not None else None
                split_factors[key] = _split_factors(splits, index[-1])
            factor = split_factors[key](fill_day)
            bucket = qty_events.setdefault(key, {})
            bucket[day] = bucket.get(day, 0.0) + _QTY_SIGN[side] * int(row["qty"]) * factor
            fill_price_events.setdefault(key, {})[day] = float(row["fill_price"]) / factor

    cash_delta = pd.Series(cash_events, dtype=float).reindex(index, fill_value=0.0)
    cash = initial_balance + cash_delta.cumsum()
    equity = cash.copy()
    fallbacks: dict[tuple[str, str], pd.DatetimeIndex] = {}
    missing_last_close: list[tuple[str, str]] = []

    for (market, symbol), bucket in qty_events.items():
        # 分割の比で直した株数の小数の誤差で、売り切った保有が残らないよう丸める
        holdings = pd.Series(bucket, dtype=float).reindex(index, fill_value=0.0).cumsum().round(6)
        if (holdings == 0).all():
            continue
        prices = price_loader(market, symbol)
        if prices is None or prices.empty:
            aligned = pd.Series(float("nan"), index=index)
        else:
            prices = prices.sort_index()
            aligned = prices.reindex(prices.index.union(index)).ffill().reindex(index)
        # 最後の日（祝日なら次の営業日）の終値がまだ無い保有は、評価が古い終値のままになる
        if holdings.iloc[-1] != 0 and (
            prices is None or prices.empty or prices.index.max() < index[-1]
        ):
            missing_last_close.append((market, symbol))
        held = holdings != 0
        missing = held & aligned.isna()
        if missing.any():
            fill_prices = pd.Series(fill_price_events[(market, symbol)], dtype=float)
            aligned = aligned.fillna(fill_prices.reindex(index).ffill())
            fallbacks[(market, symbol)] = index[missing]
            logger.warning(
                "終値が分からない保有日を約定の値段で評価: %s/%s (%d 日)",
                market,
                symbol,
                int(missing.sum()),
            )
        equity += (holdings * aligned).where(held, 0.0)

    return equity, fallbacks, missing_last_close


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
        split_loader=_default_split_loader,
    )


def _fetch_benchmark_return(
    fetcher: BenchmarkFetcher, ticker: str, base_day: pd.Timestamp, end: str
) -> Optional[float]:
    """base_day 以前の最後の終値から end の前日までのベンチマークの騰落率を返す。

    紙の売買の評価額は base_day が休場なら前の営業日の終値で評価するため、ベンチマークも
    同じ日の終値を起点にする（#1076）。fetcher は start より後の最初の行を起点にするので、
    返った起点の日が start と一致するまで営業日を1日ずつさかのぼる。
    取得できなければ None（週の道は止めない）。
    """
    for back in range(_BENCHMARK_LOOKBACK_BDAYS + 1):
        start = (base_day - pd.offsets.BDay(back)).date().isoformat()
        try:
            result = fetcher(ticker, start, end)
        except Exception:
            logger.warning("ベンチマーク取得失敗のため null にする: %s", ticker, exc_info=True)
            return None
        value = result.get("total_return")
        if value is None:
            # 取得が空・行が2つ未満・取得の形の誤りは、fetcher が None を返すだけなので理由を残す（#919）
            logger.warning(
                "ベンチマークの騰落率が取れないため null にする: %s (%s〜%s の前日まで)",
                ticker,
                start,
                end,
            )
            return None
        first_day = result.get("start")
        if first_day is None or pd.Timestamp(first_day) <= pd.Timestamp(start):
            return float(value)
    logger.warning(
        "ベンチマークの起点の終値が %s 以前に見つからないため null にする: %s",
        base_day.date().isoformat(),
        ticker,
    )
    return None


def compute_weekly_results(
    orders: pd.DataFrame,
    price_loader: Callable[[str, str], Optional[pd.Series]],
    initial_balance: float,
    as_of: date,
    benchmark_fetcher: BenchmarkFetcher,
    split_loader: Optional[SplitLoader] = None,
) -> dict[str, Any]:
    """as_of より前に終わった直近の ISO 週（月〜金）の、市場ごとの紙の売買の成績を返す（純粋ロジック）。

    as_of は日本時間の日付。金曜の当日はその週の引けがまだのため、前の週を対象にする（#919）。
    週の騰落率は前週の最終営業日の評価額から対象の週の金曜の評価額まで。
    ベンチマークも同じ期間（前週の最終営業日の終値 → 対象の週の終値）で比べる。

    Returns:
        {"week", "start", "end", "markets": {market: {weekly_return, benchmark_ticker,
        benchmark_return, excess_return, fills, valued_at_fill_price, partial_week,
        missing_week_end_close}}}。取れない値は None。
        valued_at_fill_price は、週の中に終値が分からず約定の値段で評価した日がある銘柄。
        missing_week_end_close は、週の終わりに保有しているのに金曜以降の終値がまだ無い銘柄で、
        1つでもあれば partial_week を true にする（週の終わりの評価が古い終値のままの部分の週）
    """
    day = pd.Timestamp(as_of)
    # 直前の金曜（as_of が金曜なら1週前）
    week_end = day - pd.Timedelta(days=(day.weekday() - 4) % 7 or 7)
    week_start = week_end - pd.Timedelta(days=4)
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
        valued_at_fill_price: list[str] = []
        missing_week_end_close: list[str] = []
        if not market_orders.empty:
            in_week = market_orders["filled_at"].between(week_start, week_end)
            fills = int(in_week.sum())
            first_fill = market_orders["filled_at"].min()
            index = pd.bdate_range(start=min(first_fill, base_day), end=week_end)
            equity, fallbacks, missing = _build_equity_with_fallbacks(
                market_orders, price_loader, initial_balance, index, split_loader
            )
            valued_at_fill_price = sorted(
                symbol
                for (_, symbol), days in fallbacks.items()
                if ((days >= base_day) & (days <= week_end)).any()
            )
            missing_week_end_close = sorted(symbol for _, symbol in missing)
            if missing_week_end_close:
                logger.warning(
                    "週の終わり %s の終値がまだ無い保有があるため部分の週として書く: %s %s",
                    week_end.date().isoformat(),
                    market,
                    missing_week_end_close,
                )
            base_value = float(equity.loc[base_day])
            end_value = float(equity.loc[week_end])
            weekly_return = end_value / base_value - 1 if base_value else None

        benchmark_return = _fetch_benchmark_return(
            benchmark_fetcher,
            ticker,
            base_day,
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
            "valued_at_fill_price": valued_at_fill_price,
            "partial_week": bool(missing_week_end_close),
            "missing_week_end_close": missing_week_end_close,
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
        split_loader=_default_split_loader,
    )
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{result['week']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("紙の売買の週の成績を書いた: %s", path)
    return path
