"""長期コホート・バックテストの価格ヘルパー。"""

from __future__ import annotations

from bisect import bisect_left

import pandas as pd

from src.utils.db.market_data import load_raw_closes
from src.utils.logger import get_logger

logger = get_logger(__name__)

# rescreen_freq -> pandas DateOffset（リスクリーン間隔）
FREQ_OFFSETS: dict[str, pd.DateOffset] = {
    "weekly": pd.DateOffset(weeks=1),
    "monthly": pd.DateOffset(months=1),
    "quarterly": pd.DateOffset(months=3),
    "yearly": pd.DateOffset(years=1),
    "annual": pd.DateOffset(years=1),
}


def load_price_map(market: str, end: str) -> dict[str, pd.DataFrame]:
    """market の全銘柄について date / Close を持つ価格系列を読み込む。

    end までの切り詰めは SQL 側で行う（バックテスト窓外を評価しないため）。
    銘柄ごとに 1 クエリ投げると N+1 になるので一括読み出しを使う。
    """
    raw = load_raw_closes(market, end_date=end)
    if raw.empty:
        return {}

    normalized = pd.DataFrame(
        {
            "symbol": raw["symbol"].to_numpy(),
            "date": pd.to_datetime(raw["ts"]).dt.strftime("%Y-%m-%d"),
            "Close": raw["close"].astype(float).to_numpy(),
        }
    )
    price_map: dict[str, pd.DataFrame] = {}
    for symbol, group in normalized.groupby("symbol", sort=False):
        df = group[["date", "Close"]].sort_values("date").reset_index(drop=True)
        if not df.empty:
            price_map[str(symbol)] = df
    return price_map


def build_calendar(price_map: dict[str, pd.DataFrame], start: str, end: str) -> list[str]:
    """全銘柄の取引日の和集合を [start, end] で昇順に並べたカレンダーを作る。"""
    dates: set[str] = set()
    for df in price_map.values():
        dates.update(df["date"].tolist())
    cal = sorted(d for d in dates if start <= d <= end)
    return cal


def make_rescreen_dates(calendar: list[str], start: str, freq: str) -> list[str]:
    """リスクリーン日リストを作る。

    start から freq 間隔でターゲット日を生成し、各ターゲット日以降で最初に存在する
    取引日に丸める。カレンダー外（末尾超過）のターゲットは無視する。
    """
    if not calendar:
        return []
    offset = FREQ_OFFSETS.get(freq, FREQ_OFFSETS["quarterly"])
    last = pd.Timestamp(calendar[-1])
    target = pd.Timestamp(max(start, calendar[0]))

    out: list[str] = []
    while target <= last:
        target_str = target.strftime("%Y-%m-%d")
        # calendar は昇順なので二分探索で「target 以降の最初の取引日」を引く。
        i = bisect_left(calendar, target_str)
        if i < len(calendar):
            nxt = calendar[i]
            # 長期休場を跨ぐと隣接ターゲットが同じ取引日に丸まりうる。target は
            # 単調増加ゆえ nxt も単調非減少なので、直前の採用分とだけ比較すればよい。
            if not out or out[-1] != nxt:
                out.append(nxt)
        target = target + offset
    return out


def close_lookups(
    price_map: dict[str, pd.DataFrame], calendar: list[str]
) -> dict[str, dict[str, float]]:
    """銘柄ごとに calendar 上で前方補完した date->Close を作る（保有評価用）。"""
    lookups: dict[str, dict[str, float]] = {}
    for symbol, df in price_map.items():
        s = df.set_index("date")["Close"].astype(float)
        s = s.reindex(calendar).ffill()
        lookups[symbol] = {str(k): float(v) for k, v in s.to_dict().items()}
    return lookups
