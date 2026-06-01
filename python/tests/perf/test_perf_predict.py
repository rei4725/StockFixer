"""
パフォーマンスベンチマーク: predict_single_stock

IO をモックし、特徴量計算 + モデル推論の実行時間を計測する。
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


def _make_ohlcv(periods: int = 252) -> pd.DataFrame:
    rng = np.random.default_rng(seed=1)
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


@patch("src.prediction.predict_single.load_model_weights")
@patch("src.prediction.predict_single.ModelManager")
@patch("src.prediction.predict_single.yf")
@patch("src.prediction.predict_single.get_market_data_port")
@patch("src.prediction.predict_single.os.path.exists")
@patch("src.prediction.predict_single.get_models_subdir")
def test_perf_predict_single(
    mock_subdir,
    mock_exists,
    mock_get_port,
    mock_yf,
    mock_mm_cls,
    mock_load_weights,
    record_benchmark,
    tmp_path,
):
    from src.prediction.predict_single import predict_single_stock

    mock_subdir.return_value = str(tmp_path)
    mock_exists.return_value = True

    ohlcv = _make_ohlcv(252)
    mock_port = MagicMock()
    mock_port.get_stock_data.return_value = ohlcv
    mock_port.add_technical_indicators.return_value = ohlcv
    mock_port.create_basic_lag_features.return_value = (ohlcv.iloc[:, :3], pd.Series(dtype=float))
    mock_port.fetch_cross_asset_features.return_value = None
    mock_get_port.return_value = mock_port

    hist_df = pd.DataFrame(
        {"Close": [ohlcv["Close"].iloc[-1]]},
        index=pd.date_range("2024-01-01", periods=1),
    )
    ticker_obj = MagicMock()
    ticker_obj.history.return_value = hist_df
    mock_yf.Ticker.return_value = ticker_obj

    mock_load_weights.return_value = [1.0]

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.01])
    mock_mm = MagicMock()
    mock_mm.load_model.return_value = mock_model
    mock_mm_cls.return_value = mock_mm

    start = time.perf_counter()
    result = predict_single_stock("us", "AAPL", model_types=["StockXGBoostModel.joblib"])
    elapsed = time.perf_counter() - start

    record_benchmark("predict_single_stock_252rows", elapsed)
    assert result is not None
    assert elapsed < 60.0, f"predict_single_stock が遅すぎる: {elapsed:.2f}s"
