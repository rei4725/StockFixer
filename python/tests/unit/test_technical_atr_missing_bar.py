"""高値・安値・終値の欠けた足1本で ATR が固まらないことを確かめる（#1240）。"""

import numpy as np
import pandas as pd
import pytest
import ta

from src.market_data.technical import add_technical_indicators
from src.rule_engine.rules.technical import VolatilityBreakoutRule

_ATR_WINDOW = 14


def _bars(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    spread = rng.uniform(0.5, 3.0, n)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + spread,
            "Low": close - spread,
            "Close": close,
            "Volume": rng.integers(1000, 2000, n).astype(float),
        },
        index=pd.bdate_range("2024-01-01", periods=n),
    )


def _with_missing_bar(df: pd.DataFrame, pos: int) -> pd.DataFrame:
    broken = df.copy()
    broken.iloc[pos, broken.columns.get_indexer(["High", "Low", "Close"])] = np.nan
    return broken


def _ta_atr(df: pd.DataFrame) -> pd.Series:
    return ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=_ATR_WINDOW, fillna=True
    ).average_true_range()


def test_atr_after_missing_bar_matches_atr_of_complete_bars():
    df = _bars()
    missing = 40
    broken = _with_missing_bar(df, missing)

    atr = add_technical_indicators(broken)["atr"]

    expected = _ta_atr(broken.drop(index=broken.index[missing]))
    after = broken.index[missing + 1 :]
    pd.testing.assert_series_equal(
        atr.loc[after], expected.loc[after], check_names=False, check_freq=False
    )
    # 欠けた足の手前の値に固まらない
    assert atr.iloc[-1] != pytest.approx(atr.iloc[missing - 1])
    assert atr.loc[after].nunique() > 1


def test_atr_on_missing_bar_keeps_previous_value():
    df = _bars()
    missing = 40
    atr = add_technical_indicators(_with_missing_bar(df, missing))["atr"]

    # 未来の足を使わず、1本手前の値を引き継ぐ
    assert atr.iloc[missing] == pytest.approx(atr.iloc[missing - 1])
    assert not atr.isna().any()


def test_atr_without_missing_bar_is_unchanged():
    df = _bars()
    atr = add_technical_indicators(df)["atr"]
    pd.testing.assert_series_equal(atr, _ta_atr(df), check_names=False)


@pytest.mark.parametrize(
    "col",
    [
        "macd",
        "macd_signal",
        "ema_fast",
        "ema_slow",
        "rsi",
        "bb_middle",
        "stoch_k",
        "obv",
    ],
)
def test_other_indicators_after_missing_bar(col):
    """ほかの ta の指標は1本の欠けの後も固まらずに動き続ける（直すのは ATR だけ）。"""
    df = _bars()
    missing = 40
    out = add_technical_indicators(_with_missing_bar(df, missing))[col]
    tail = out.iloc[missing + 1 :]
    assert not tail.isna().any()
    assert tail.nunique() == len(tail)


def test_volatility_breakout_sell_line_uses_current_atr_after_missing_bar():
    n = 120
    missing = 30
    close = np.full(n, 100.0)
    # 欠けた足の前は値幅10の荒い足、後は値幅1の静かな足
    spread = np.where(np.arange(n) < missing, 5.0, 0.5)
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close + spread,
            "Low": close - spread,
            "Close": close,
            "Volume": np.full(n, 1000.0),
        },
        index=pd.bdate_range("2024-01-01", periods=n),
    )
    # 最後の足で5下げる。今の ATR（約1）なら売りの線を割り、固まった ATR（約10）なら割らない
    df.iloc[-1, df.columns.get_indexer(["Close", "Low"])] = [95.0, 94.5]
    broken = _with_missing_bar(df, missing)

    enriched = add_technical_indicators(broken)
    signal = VolatilityBreakoutRule().generate_signal(enriched)

    prev_atr = enriched["atr"].iloc[-2]
    expected_prev_atr = _ta_atr(broken.drop(index=broken.index[missing])).iloc[-2]
    assert prev_atr == pytest.approx(expected_prev_atr)
    assert prev_atr < 2.0
    assert signal.iloc[-1] == -1
