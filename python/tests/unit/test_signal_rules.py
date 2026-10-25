"""domain の売買シグナルの純粋関数と、utils の SignalGenerator が同じ出力を返すこと（2026-10-25 設計書 §4）。"""

import numpy as np
import pandas as pd
import pytest

from config.settings import VOLUME_FILTER_MULTIPLIER, VOLUME_FILTER_WINDOW_DAYS
from src.domain.signal_rules import generate_trade_signals
from src.utils.signal_generator import SignalGenerator


def _frame(with_rsi: bool, with_volume: bool) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=60, freq="B")
    cols = {}
    if with_rsi:
        cols["RSI"] = rng.uniform(10, 90, len(idx))
    if with_volume:
        cols["Volume"] = rng.integers(1_000, 10_000, len(idx)).astype(float)
    return pd.DataFrame(cols, index=idx)


@pytest.mark.parametrize("with_rsi", [True, False])
@pytest.mark.parametrize("with_volume", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
def test_pure_function_matches_signal_generator(with_rsi, with_volume, with_std):
    data = _frame(with_rsi, with_volume)
    rng = np.random.default_rng(1)
    prediction = pd.Series(rng.normal(0, 0.01, len(data)), index=data.index)
    rolling_std = prediction.rolling(5).std() if with_std else None

    expected = SignalGenerator(base_threshold=0.005).generate_signal(data, prediction, rolling_std)
    actual = generate_trade_signals(
        data, prediction, 0.005, VOLUME_FILTER_WINDOW_DAYS, VOLUME_FILTER_MULTIPLIER, rolling_std
    )
    pd.testing.assert_series_equal(actual, expected)


def test_zero_average_std_falls_back_to_fixed_threshold():
    data = _frame(False, False)
    prediction = pd.Series(0.006, index=data.index)
    rolling_std = pd.Series(0.0, index=data.index)
    result = generate_trade_signals(data, prediction, 0.005, 20, 1.5, rolling_std)
    assert (result == "Buy").all()
