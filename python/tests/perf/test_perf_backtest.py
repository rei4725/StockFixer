"""
パフォーマンスベンチマーク: BacktestPipeline (Backtester)

IO をモックし、バックテストのシミュレーション処理時間を計測する。
"""

import time
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.backtest.backtester import Backtester


def _make_ohlcv_with_ta(periods: int = 252) -> pd.DataFrame:
    from src.market_data.technical import add_technical_indicators

    rng = np.random.default_rng(seed=2)
    dates = pd.date_range("2023-01-01", periods=periods, freq="B")
    close = 100.0 + np.cumsum(rng.uniform(-1.0, 1.0, periods))
    df = pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": rng.integers(1000, 5000, periods).astype(float),
        },
        index=dates,
    )
    return add_technical_indicators(df)


def test_perf_backtester_run(record_benchmark):
    df = _make_ohlcv_with_ta(252)

    mock_data_loader = MagicMock()
    mock_data_loader.get_stock_data_auto.return_value = df

    mock_model = MagicMock()
    mock_model.predict.side_effect = lambda X: np.random.uniform(-0.02, 0.02, len(X))

    mock_model_manager = MagicMock()
    mock_model_manager.get_model.return_value = mock_model

    mock_signal_gen = MagicMock()

    backtester = Backtester(
        model_manager=mock_model_manager,
        signal_generator=mock_signal_gen,
        data_loader=mock_data_loader,
        start_date="2023-01-01",
        end_date="2023-12-31",
        market="us",
        symbol="AAPL",
    )

    start = time.perf_counter()
    result_df, metrics = backtester.run(model_name="test_model")
    elapsed = time.perf_counter() - start

    record_benchmark("backtester_run_252rows", elapsed)
    assert result_df is not None
    assert isinstance(metrics, dict)
    assert elapsed < 60.0, f"Backtester.run が遅すぎる: {elapsed:.2f}s"
