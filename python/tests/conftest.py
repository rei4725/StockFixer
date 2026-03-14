"""
pytest 共通 fixture と conftest

Unit Test / Integration Test で共有する fixture を定義。
"""
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ============================================
# データ生成 Helper
# ============================================


@pytest.fixture
def sample_price_df():
    """サンプル株価 DataFrame を生成する fixture"""
    return pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104],
            "High": [102, 103, 104, 105, 106],
            "Low": [99, 100, 101, 102, 103],
            "Close": [101, 102, 103, 104, 105],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        },
        index=pd.date_range("2024-01-01", periods=5, freq="B"),
    )


@pytest.fixture
def sample_features_df():
    """サンプル特徴量 DataFrame を生成する fixture"""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    return pd.DataFrame(
        {
            "Close": [100 + i for i in range(10)],
            "Close_lag1": [0] + [100 + i for i in range(9)],
            "Close_lag2": [0, 0] + [100 + i for i in range(8)],
            "Volume": [1000 + i * 100 for i in range(10)],
            "SMA_5": [100 + i * 0.5 for i in range(10)],
            "RSI": [50 + i * 2 for i in range(10)],
            "symbol": ["7203"] * 10,
            "market": ["jp"] * 10,
        },
        index=dates,
    )


@pytest.fixture
def sample_signal_series(sample_price_df):
    """サンプルシグナル Series (1/0/-1) を生成する fixture"""
    return pd.Series(
        [1, 0, 0, -1, 0],  # Buy → Hold → Hold → Sell → Hold
        index=sample_price_df.index,
    )


# ============================================
# Mock オブジェクト Factory
# ============================================


@pytest.fixture
def mock_model_manager():
    """Fixture providing a mocked ModelManager."""
    mock = MagicMock()
    # predict は Series を返す
    mock.predict_with_model.return_value = [1.0, 0.5, -0.5, -1.0, 0.2]
    return mock


@pytest.fixture
def mock_signal_generator():
    """Fixture providing a mocked SignalGenerator."""
    mock = MagicMock()
    mock.generate.return_value = pd.Series([1, 0, -1, 0, 0])
    return mock


@pytest.fixture
def mock_data_loader():
    """Fixture providing a mocked DataLoader."""
    mock = MagicMock()
    # get_stock_data_auto は DataFrame を返す
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    mock.get_stock_data_auto.return_value = pd.DataFrame(
        {"Close": [100 + i for i in range(10)], "Volume": [1000 + i * 100 for i in range(10)]},
        index=dates,
    )
    return mock


# ============================================
# Backtester Factory
# ============================================


@pytest.fixture
def backtester_with_mocks(mock_model_manager, mock_signal_generator, mock_data_loader):
    """Backtester オブジェクトをモック依存で生成する fixture"""
    from src.backtest.backtester import Backtester

    return Backtester(
        model_manager=mock_model_manager,
        signal_generator=mock_signal_generator,
        data_loader=mock_data_loader,
        start_date="2024-01-01",
        end_date="2024-01-31",
        market="jp",
        symbol="7203",
        initial_cash=1_000_000,
        fee_rate=0.001,
        slippage=0.0,
    )


# ============================================
# 長期データ Fixture
# ============================================


@pytest.fixture
def sample_price_df_long():
    """テクニカル指標計算に十分な行数（60行）のサンプル株価 DataFrame"""
    import numpy as np

    rng = np.random.default_rng(seed=42)
    dates = pd.date_range("2023-01-01", periods=60, freq="B")
    close = 100.0 + np.cumsum(rng.uniform(-1.0, 1.0, 60))
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": rng.integers(1000, 5000, 60),
        },
        index=dates,
    )


@pytest.fixture(params=["jp", "us", "nasdaq"])
def market(request):
    """マーケット種別をパラメータ化する fixture"""
    return request.param


# ============================================
# 環境チェック Marker
# ============================================


@pytest.fixture(scope="session")
def has_xgboost():
    """Check if XGBoost is available."""
    try:
        import xgboost  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def has_duckdb():
    """Check if DuckDB is available."""
    try:
        import duckdb  # noqa: F401

        return True
    except ImportError:
        return False
