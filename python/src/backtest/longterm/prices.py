"""長期コホート・バックテストの価格ヘルパー。"""

from __future__ import annotations

from bisect import bisect_left

import pandas as pd

from src.utils.db.market_data import load_raw_closes

# rescreen_freq -> pandas DateOffset（リスクリーン間隔）
FREQ_OFFSETS: dict[str, pd.DateOffset] = {
    "weekly": pd.DateOffset(weeks=1),
    "monthly": pd.DateOffset(months=1),
    "quarterly": pd.DateOffset(months=3),
    "yearly": pd.DateOffset(years=1),
    "annual": pd.DateOffset(years=1),
}


AVG_VOLUME_WINDOW = 20


def load_price_map(market: str, end: str) -> dict[str, pd.DataFrame]:
    """market の全銘柄について date / Close / AvgVolume を持つ価格系列を読み込む。

    end までの切り詰めは SQL 側で行う（バックテスト窓外を評価しないため）。
    銘柄ごとに 1 クエリ投げると N+1 になるので一括読み出しを使う。

    AvgVolume は約定の日より前の 20 本の平均出来高（rolling(20).mean().shift(1)、
    約定の日を含めない。ポートフォリオのバックテストと同じ）。動的スリッページの
    volume に渡す。先頭の日や出来高の無い行は 0（フラットのみ）。
    """
    raw = load_raw_closes(market, end_date=end, with_volume=True)
    if raw.empty:
        return {}

    volume = raw["volume"] if "volume" in raw.columns else pd.Series(0.0, index=raw.index)
    normalized = pd.DataFrame(
        {
            "symbol": raw["symbol"].to_numpy(),
            "date": pd.to_datetime(raw["ts"]).dt.strftime("%Y-%m-%d"),
            "Close": raw["close"].astype(float).to_numpy(),
            "Volume": volume.astype(float).fillna(0.0).to_numpy(),
        }
    )
    price_map: dict[str, pd.DataFrame] = {}
    for symbol, group in normalized.groupby("symbol", sort=False):
        df = group[["date", "Close", "Volume"]].sort_values("date").reset_index(drop=True)
        if df.empty:
            continue
        avg = df["Volume"].rolling(AVG_VOLUME_WINDOW, min_periods=1).mean().shift(1)
        df["AvgVolume"] = avg.fillna(0.0)
        price_map[str(symbol)] = df[["date", "Close", "AvgVolume"]]
    return price_map


def exec_avg_volume(price_map: dict[str, pd.DataFrame], symbol: str, date: str) -> float:
    """symbol の date の約定に使う平均出来高を返す。列や行が無ければ 0（フラットのみ）。"""
    df = price_map.get(symbol)
    if df is None or "AvgVolume" not in df.columns:
        return 0.0
    row = df.loc[df["date"] == date, "AvgVolume"]
    return float(row.iloc[0]) if not row.empty else 0.0


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
