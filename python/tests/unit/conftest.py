# Unit test専用 fixture。外部DB・API依存は含まない。
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ============================================
# Discord 実送信ガード（unit テスト全体に適用）
# ============================================


@pytest.fixture(autouse=True)
def _block_discord_http(monkeypatch):
    """Block real Discord HTTP calls from all unit tests.

    DISCORD_WEBHOOK_URL を除去することで _post_webhook が早期 return し、
    どのテストからも実際の webhook エンドポイントに到達しなくなる。
    Discord 送信の有無を検証したいテストは @patch で個別に制御する。
    """
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)


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
