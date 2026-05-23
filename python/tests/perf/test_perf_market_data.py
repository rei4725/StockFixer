"""
パフォーマンスベンチマーク: market_data パイプライン

特徴量計算（テクニカル指標 + ラグ特徴量）の実行時間を計測する。
外部 IO はなく、CPU バウンドな純粋計算のみを対象とする。
"""

import time

import numpy as np
import pandas as pd
import pytest

from src.market_data.technical import add_technical_indicators, create_basic_lag_features


def _make_ohlcv(periods: int = 252) -> pd.DataFrame:
    rng = np.random.default_rng(seed=0)
    dates = pd.date_range("2023-01-01", periods=periods, freq="B")
    close = 100.0 + np.cumsum(rng.uniform(-1.0, 1.0, periods))
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": rng.integers(1000, 5000, periods).astype(float),
        },
        index=dates,
    )


def test_perf_add_technical_indicators(record_benchmark):
    df = _make_ohlcv(252)

    start = time.perf_counter()
    result = add_technical_indicators(df)
    elapsed = time.perf_counter() - start

    record_benchmark("add_technical_indicators_252rows", elapsed)
    assert result is not None and not result.empty
    assert elapsed < 30.0, f"add_technical_indicators が遅すぎる: {elapsed:.2f}s"


def test_perf_create_basic_lag_features(record_benchmark):
    df = _make_ohlcv(252)
    df_with_ta = add_technical_indicators(df)

    start = time.perf_counter()
    X, y = create_basic_lag_features(df_with_ta)
    elapsed = time.perf_counter() - start

    record_benchmark("create_basic_lag_features_252rows", elapsed)
    assert len(X) > 0
    assert elapsed < 30.0, f"create_basic_lag_features が遅すぎる: {elapsed:.2f}s"


def test_perf_full_feature_pipeline(record_benchmark):
    """テクニカル指標 + ラグ特徴量の一連の処理を計測する"""
    df = _make_ohlcv(252)

    start = time.perf_counter()
    df_ta = add_technical_indicators(df)
    X, y = create_basic_lag_features(df_ta)
    elapsed = time.perf_counter() - start

    record_benchmark("full_feature_pipeline_252rows", elapsed)
    assert len(X) > 0 and len(y) > 0
    assert elapsed < 60.0, f"特徴量パイプライン全体が遅すぎる: {elapsed:.2f}s"
