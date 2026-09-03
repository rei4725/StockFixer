"""レジームレバレッジ戦略で使う指標計算(ATR・200日線)。

trading-strategy/backtest/backtest.py の wilder_atr、
backtest_regime.py の compute_regime_indicators と同じ計算式を使う
(バックテスト結果との整合性を保つため)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_REGIME_MA = 200
_ATR_PERIOD = 14


def wilder_atr(df: pd.DataFrame, period: int = _ATR_PERIOD) -> pd.Series:
    """Wilderのスムージング法によるATRを計算する。"""
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.copy()
    atr.iloc[:period] = np.nan
    atr.iloc[period - 1] = tr.iloc[0:period].mean()
    for i in range(period, len(tr)):
        atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + tr.iloc[i]) / period
    return atr


def build_weekly_frame(daily_df: pd.DataFrame) -> pd.DataFrame:
    """日足dfにATR14・200日線を追加する(週足化はしない。呼び出し側が最終行=直近営業日
    を使う。7章の判定は「その週の金曜終値」だが、日次ジョブ実行時点では当該週がまだ
    確定していないため、呼び出し側で週の最終営業日かどうかを判定する)。
    """
    df = daily_df.copy()
    df["ATR14"] = wilder_atr(df, _ATR_PERIOD)
    df["MA200"] = df["Close"].rolling(_REGIME_MA).mean()
    return df
