"""
pytest 共通 fixture と conftest

Unit Test / Integration Test で共有する fixture を定義。
"""
import pytest
import pandas as pd
from unittest.mock import MagicMock
from datetime import datetime, timedelta


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
    """ModelManager のモックを生成する fixture"""
    mock = MagicMock()
    # predict は Series を返す
    mock.predict_with_model.return_value = [1.0, 0.5, -0.5, -1.0, 0.2]
    return mock


@pytest.fixture
def mock_signal_generator():
    """SignalGenerator のモックを生成する fixture"""
    mock = MagicMock()
    mock.generate.return_value = pd.Series([1, 0, -1, 0, 0])
    return mock


@pytest.fixture
def mock_data_loader():
    """DataLoader のモックを生成する fixture"""
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
# 環境チェック Marker
# ============================================

pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration


@pytest.fixture(scope="session")
def has_xgboost():
    """XGBoost が利用可能かチェック"""
    try:
        import xgboost
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def has_duckdb():
    """DuckDB が利用可能かチェック"""
    try:
        import duckdb
        return True
    except ImportError:
        return False
