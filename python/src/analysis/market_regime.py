from __future__ import annotations

from typing import Literal

import pandas as pd

MarketRegime = Literal["bull", "bear", "range"]


def get_market_regime(
    df: pd.DataFrame,
    ema_window: int = 200,
    atr_window: int = 14,
    vol_high_quantile: float = 0.67,
) -> pd.Series:
    """
    市場レジームを bull / bear / range に分類する。

    Close と EMA の位置関係で方向感を、ATR 相当の変動幅で高ボラ局面を判定する。
    高ボラ日は方向に関係なく "range" を優先する。
    """
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=str, index=getattr(df, "index", None))

    close = pd.Series(df["Close"].values, index=df.index, dtype=float)
    actual_window = min(ema_window, max(1, len(df) - 1))
    ema = close.ewm(span=actual_window, adjust=False).mean()

    if "High" in df.columns and "Low" in df.columns:
        high = pd.Series(df["High"].values, index=df.index, dtype=float)
        low = pd.Series(df["Low"].values, index=df.index, dtype=float)
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
    else:
        tr = close.pct_change().abs()

    atr = tr.rolling(window=atr_window, min_periods=1).mean()
    vol_threshold = float(atr.quantile(vol_high_quantile))

    regime = pd.Series("bear", index=df.index, dtype=str)
    regime[close > ema] = "bull"
    regime[atr >= vol_threshold] = "range"
    return regime
