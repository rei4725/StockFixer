"""
分割の段差の判定（#1492・#2031・#2035）

前日比の終値が極端な足を段差とし、比が単純な分割・併合の比に近く最新の足に近い段差を
記録待ちの分割とみなす。market_data の取り直し（split_refetch）と prediction の採点が
同じ判定を使うため、shared kernel に置く。
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# 前日比の終値がこの範囲の外の足を段差とみなす
PRICE_STEP_MIN_RATIO = 0.55
PRICE_STEP_MAX_RATIO = 1.8
# 取り直した後に残る段差は、この営業日数以内に分割の記録があれば分割の残りとみなす（#1492）
SPLIT_RECORD_NEAR_WEEKDAYS = 5
# 段差の比と分割の比（またはその逆数）の積がこの割合を超えて 1 からずれなければ、比が近いとみなす
SPLIT_RATIO_TOLERANCE = 0.05


def day_index(index: pd.Index) -> pd.DatetimeIndex:
    """取引所の日付（タイムゾーン無しの 0 時）にそろえた index を返す。"""
    days = pd.DatetimeIndex(pd.to_datetime(index))
    if days.tz is not None:
        days = days.tz_localize(None)
    return days.normalize()


def find_price_steps(df: Optional[pd.DataFrame]) -> List[Tuple[str, float]]:
    """前日比の終値が極端な足の (日付, 比) を返す。"""
    if df is None or df.empty or "Close" not in df.columns:
        return []
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    ratios = (close / close.shift(1)).iloc[1:]
    steps = ratios[(ratios < PRICE_STEP_MIN_RATIO) | (ratios > PRICE_STEP_MAX_RATIO)]
    days = day_index(steps.index)
    return [(day.strftime("%Y-%m-%d"), float(r)) for day, r in zip(days, steps)]


def is_pending_split(step_day: np.datetime64, ratio: float, latest_day: np.datetime64) -> bool:
    """
    記録がまだ無い分割・併合とみなす段差か（#2031）。

    yfinance は分割の記録と過去の足の調整を数日遅れて入れることがある。比が単純な分割・併合の比
    （1/n か n、n=2〜10）に SPLIT_RATIO_TOLERANCE の内で近く、段差の日が最新の足から
    SPLIT_RECORD_NEAR_WEEKDAYS 営業日以内なら、記録待ちの分割とする。それを過ぎても記録が
    来なければ本当の値動きとする。
    """
    if abs(int(np.busday_count(step_day, latest_day))) > SPLIT_RECORD_NEAR_WEEKDAYS:
        return False
    return any(
        min(abs(ratio * n - 1.0), abs(ratio / n - 1.0)) <= SPLIT_RATIO_TOLERANCE
        for n in range(2, 11)
    )


def pending_split_steps(
    df: Optional[pd.DataFrame], latest: pd.Timestamp
) -> List[Tuple[str, float]]:
    """df の足の段差のうち、最新の足の日 latest から見て記録待ちの分割とみなすものを返す（#2035）。"""
    latest_day = day_index(pd.DatetimeIndex([latest]))[0].to_datetime64().astype("datetime64[D]")
    return [
        (day, ratio)
        for day, ratio in find_price_steps(df)
        if is_pending_split(np.datetime64(day, "D"), ratio, latest_day)
    ]
