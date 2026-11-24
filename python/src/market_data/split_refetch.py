"""
株式分割で値段の単位がずれた足の全期間の取り直し（#817・#1386）

yfinance は分割の前の足も調整し直して返すが、DB の古い足は分割前の値段のまま。
差分とつなぐと分割の比の段差になるため、全期間を取り直す。

分割は差分の足の Stock Splits の列だけでは見落とす（PR 892 より前の分割・
取得が止まっていた間の分割・yfinance が行を返さない分割）。そのため、
差分を DB の直近の足と重ねて取って終値を突き合わせ、取り直した後も
前日比が極端な足が残り、その近くに分割の記録がある銘柄は予想から外す
（記録の無い段差は本当の値動きとみなす。#1492）。
取り直しは取得の窓より前の DB の足も含める（#1392）。

重ねて取った足は、DB に残った途中の足（出来高が本当よりずっと少ない足）を直すのにも使う（#2140）。
"""

import threading
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.domain.split_steps import (  # noqa: F401  段差の判定は domain から再輸出する
    PRICE_STEP_MAX_RATIO,
    PRICE_STEP_MIN_RATIO,
    SPLIT_RATIO_TOLERANCE,
    SPLIT_RECORD_NEAR_WEEKDAYS,
    day_index,
    find_price_steps,
    is_pending_split,
)
from src.market_data.base import DataSourceBase
from src.utils.logger import get_logger

logger = get_logger(__name__)

_day_index = day_index
_pending_split = is_pending_split

# 差分を取るときに重ねる DB の直近の足の本数
OVERLAP_BARS = 5
# 重なる足の終値の比（yfinance / DB）がこれを超えて 1 からずれたら調整し直されたとみなす。
# 配当の調整による小さなずれはこの中に収まる
OVERLAP_CLOSE_TOLERANCE = 0.03
# 重なる足の出来高の比（yfinance / DB）がこれ以上なら、DB の足を途中の足とみなす（#2140）。
# 引け後のふつうの訂正（数％）には当たらず、出来高が本当の 15% の途中の足（約 6.7 倍）には当たる。
# 当たり過ぎると直近の数本だけが配当の調整後の値になるため、線は余裕を持たせて 2 倍
# （DB の出来高が本当の半分以下）に置く。終値は配当の調整で動くので判定に使わない
PARTIAL_BAR_VOLUME_RATIO = 2.0
_PRICE_COLUMNS = ("Open", "High", "Low", "Close")

# この回で直した足の日（market, symbol ごと）。取得は並列なので lock で守る
_repaired_days: Dict[Tuple[str, str], List[str]] = {}
_repairs_lock = threading.Lock()


def stock_split_ratios(fresh_data: Optional[pd.DataFrame]) -> list:
    """差分の足の Stock Splits の列から、0 でない分割の比を返す。列が無ければ空。"""
    if fresh_data is None or "Stock Splits" not in fresh_data.columns:
        return []
    splits = pd.to_numeric(fresh_data["Stock Splits"], errors="coerce").fillna(0.0)
    return [float(v) for v in splits[splits != 0]]


def overlap_fetch_start(db_bars: pd.DataFrame, n_bars: int = OVERLAP_BARS) -> str:
    """DB の直近 n_bars 本の最初の日（差分の取得の始まり）を YYYY-MM-DD で返す。"""
    days = _day_index(db_bars.index).sort_values()
    return days[max(len(days) - n_bars, 0)].strftime("%Y-%m-%d")


def bars_after(fresh_data: Optional[pd.DataFrame], latest: pd.Timestamp) -> Optional[pd.DataFrame]:
    """latest の日より後の足だけを返す（重ねて取った足を差分に戻す）。"""
    if fresh_data is None or fresh_data.empty:
        return fresh_data
    latest_day = _day_index(pd.DatetimeIndex([latest]))[0]
    return fresh_data[_day_index(fresh_data.index) > latest_day]


def overlap_close_mismatch(
    db_bars: pd.DataFrame,
    fresh_data: Optional[pd.DataFrame],
    tolerance: float = OVERLAP_CLOSE_TOLERANCE,
) -> Optional[float]:
    """
    同じ日の終値の比（yfinance / DB）が 1 から tolerance を超えてずれていれば、
    いちばんずれた比を返す。重なる足が無い・ずれが閾値の中なら None。
    """
    if fresh_data is None or fresh_data.empty:
        return None
    if "Close" not in db_bars.columns or "Close" not in fresh_data.columns:
        return None
    db_close = pd.Series(db_bars["Close"].to_numpy(), index=_day_index(db_bars.index))
    yf_close = pd.Series(fresh_data["Close"].to_numpy(), index=_day_index(fresh_data.index))
    db_close = db_close[~db_close.index.duplicated(keep="last")]
    yf_close = yf_close[~yf_close.index.duplicated(keep="last")]
    common = db_close.index.intersection(yf_close.index)
    ratios = (yf_close[common] / db_close[common]).replace([float("inf")], float("nan")).dropna()
    if ratios.empty:
        return None
    worst = ratios.iloc[int((ratios - 1.0).abs().argmax())]
    if abs(worst - 1.0) <= tolerance:
        return None
    return float(worst)


def partial_bar_repairs(
    db_bars: pd.DataFrame,
    fresh_data: Optional[pd.DataFrame],
    min_ratio: float = PARTIAL_BAR_VOLUME_RATIO,
) -> pd.DataFrame:
    """
    DB の側が途中の足とみなす日の、取り直した足を返す（#2140）。無ければ空の表。

    取り直した出来高が DB の出来高の min_ratio 倍以上の日だけを返す。取り直した足の
    出来高が欠け・0、四本値が欠けている日と、どちらかに同じ日が2行ある日は返さない。
    """
    needed = [*_PRICE_COLUMNS, "Volume"]
    if fresh_data is None or fresh_data.empty:
        return pd.DataFrame()
    if any(c not in db_bars.columns or c not in fresh_data.columns for c in needed):
        return fresh_data.iloc[:0]
    db_days, fresh_days = _day_index(db_bars.index), _day_index(fresh_data.index)
    db_volume = pd.Series(
        pd.to_numeric(db_bars["Volume"], errors="coerce").to_numpy(dtype=float), index=db_days
    )
    db_volume = db_volume[~db_days.duplicated(keep=False)].reindex(fresh_days).to_numpy()
    volume = pd.to_numeric(fresh_data["Volume"], errors="coerce").to_numpy(dtype=float)
    has_prices = fresh_data[list(_PRICE_COLUMNS)].notna().all(axis=1).to_numpy()
    # 欠け（NaN）との比べは False になり、新しい日（DB に無い）と欠けた出来高はここで外れる
    partial = (volume > 0) & (volume >= db_volume * min_ratio)
    return fresh_data[partial & has_prices & ~fresh_days.duplicated(keep=False)]


def replace_bars(db_bars: pd.DataFrame, repairs: pd.DataFrame) -> pd.DataFrame:
    """
    db_bars の同じ日の行を repairs の行に置き換える。

    merge_market_data は重なる日で DB の行を残すため、その後に置き換える。
    """
    kept = db_bars[~_day_index(db_bars.index).isin(_day_index(repairs.index))]
    rows = repairs.reindex(columns=db_bars.columns)
    rows.index = _day_index(repairs.index).tz_localize(pd.DatetimeIndex(db_bars.index).tz)
    merged = pd.concat([kept, rows]) if not kept.empty else rows
    merged.index.name = db_bars.index.name
    return merged.sort_index()


def record_bar_repairs(market: str, symbol: str, repairs: pd.DataFrame) -> None:
    """直した足の日を、回のまとめのログのために控える（銘柄ごとのログは出さない）。"""
    days = [str(d) for d in _day_index(repairs.index).strftime("%Y-%m-%d")]
    with _repairs_lock:
        _repaired_days[(market, symbol)] = days


def reset_bar_repairs() -> None:
    """直した足の控えを空にする（回の初めに呼ぶ）。"""
    with _repairs_lock:
        _repaired_days.clear()


def log_bar_repairs() -> None:
    """控えた銘柄の数と日付ごとの数を1行の INFO で出し、控えを空にする。無ければ出さない。"""
    with _repairs_lock:
        repaired = dict(_repaired_days)
        _repaired_days.clear()
    if not repaired:
        return
    counts = Counter(day for days in repaired.values() for day in days)
    detail = "、".join(f"{day}: {n}" for day, n in sorted(counts.items()))
    logger.info(f"途中の足を取り直した足で直す: {len(repaired)}銘柄（日付ごとの銘柄の数 {detail}）")


def steps_near_split_records(
    steps: List[Tuple[str, float]], full_data: Optional[pd.DataFrame]
) -> List[Tuple[str, float]]:
    """
    取り直した足の Stock Splits の列に照らし、分割の残りとみなす段差だけを返す（#1492）。

    取り直した足は yfinance が調整し直した1本の系列なので、残る段差は調整し損ねた分割か
    本当の値動きかのどちらか。近い日（SPLIT_RECORD_NEAR_WEEKDAYS 営業日以内）に分割の
    記録があるか、比が記録された分割の比に近い段差だけを分割の残りとする。
    記録が無くても、最新の足に近く比が単純な分割の比に近い段差は記録待ちの分割とする（#2031）。
    Stock Splits の列が無ければ見分けられないため、段差をすべて返す（今までどおり外す）。
    """
    if not steps:
        return []
    if full_data is None or "Stock Splits" not in full_data.columns:
        return list(steps)
    splits = pd.to_numeric(full_data["Stock Splits"], errors="coerce").fillna(0.0)
    splits = splits[splits > 0]
    split_days = _day_index(splits.index).values.astype("datetime64[D]")
    split_ratios = [float(v) for v in splits]
    latest_day = _day_index(full_data.index).max().to_datetime64().astype("datetime64[D]")
    kept = []
    for day, ratio in steps:
        step_day = np.datetime64(day, "D")
        near_day = any(
            abs(int(np.busday_count(min(step_day, d), max(step_day, d))))
            <= SPLIT_RECORD_NEAR_WEEKDAYS
            for d in split_days
        )
        near_ratio = any(
            min(abs(ratio * s - 1.0), abs(ratio / s - 1.0)) <= SPLIT_RATIO_TOLERANCE
            for s in split_ratios
        )
        if near_day or near_ratio or _pending_split(step_day, ratio, latest_day):
            kept.append((day, ratio))
    return kept


def split_residual_steps(
    market: str, symbol: str, df: pd.DataFrame, full_data: Optional[pd.DataFrame]
) -> List[Tuple[str, float]]:
    """取り直した後の窓の足 df に残る段差のうち、分割の残りとみなすものを返す（#1492）。"""
    left_steps = find_price_steps(df)
    steps = steps_near_split_records(left_steps, full_data)
    if len(steps) < len(left_steps):
        logger.info(
            f"分割の記録が無い段差は本当の値動きとして特徴量を作る: {market}/{symbol} "
            f"{[s for s in left_steps if s not in steps][:3]}"
        )
    return steps


def refetch_full_history(
    market: str,
    symbol: str,
    ticker: str,
    start_date: str,
    end_date: str,
    reason: str,
    data_source: Optional[DataSourceBase],
    fetch_stock_data: Callable[[str, str, str, str], Optional[pd.DataFrame]],
    load_db_bars: Optional[Callable[..., Optional[pd.DataFrame]]] = None,
) -> Optional[pd.DataFrame]:
    """
    株式分割のため全期間の足を取り直す。

    取得の窓は毎日後ろへ動くため、DB には start_date より前の足が残る。
    それも分割前の値段のまま段差になるので、取り直しの始まりを DB の最古の足の日まで広げる（#1392）。

    Args:
        reason: 取り直す理由（ログに出す。例: 分割の比・突き合わせの比）
        load_db_bars: (market, symbol, start, end) で DB の足を読む関数。None なら広げない

    Returns:
        取り直した全期間の足（start_date より前を含みうる）。取れなければ None
        （段差のある足で特徴量を作らない）。
    """
    db_oldest = _db_oldest_day(market, symbol, start_date, load_db_bars)
    fetch_start = min(start_date, db_oldest) if db_oldest else start_date
    logger.info(
        f"株式分割のため全期間を取り直す: {market}/{symbol} "
        f"（{reason}、{fetch_start}～{end_date}）"
    )
    if data_source is not None:
        full_data = data_source.get_stock_data(symbol, market, fetch_start, end_date)
    else:
        full_data = fetch_stock_data(market, ticker, fetch_start, end_date)
    if full_data is None or full_data.empty:
        logger.warning(
            f"株式分割の取り直しに失敗したため特徴量生成をスキップ: {market}/{symbol}"
            "（段差のある足で特徴量を作らない）"
        )
        return None
    first_day = _day_index(full_data.index).min().strftime("%Y-%m-%d")
    if db_oldest and first_day > db_oldest:
        logger.warning(
            f"取り直した足が DB の最古の足より新しく、{db_oldest}～{first_day} の DB の足は"
            f"分割前の値段のまま残る: {market}/{symbol}"
        )
    return full_data


def _db_oldest_day(
    market: str,
    symbol: str,
    start_date: str,
    load_db_bars: Optional[Callable[..., Optional[pd.DataFrame]]],
) -> Optional[str]:
    """DB にある start_date までの足のうち、最古の日を YYYY-MM-DD で返す。無ければ None。"""
    if load_db_bars is None:
        return None
    older = load_db_bars(market, symbol, None, start_date)
    if older is None or older.empty:
        return None
    return str(_day_index(older.index).min().strftime("%Y-%m-%d"))


def bars_from(df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    """start_date の日以後の足だけを返す（広げて取り直した足を特徴量の窓に戻す）。"""
    return df[_day_index(df.index) >= pd.Timestamp(start_date)]
