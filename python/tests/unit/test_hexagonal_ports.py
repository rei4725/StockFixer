"""ヘキサゴナルアーキテクチャ ポート・インメモリアダプターのユニットテスト"""

import pandas as pd
import pytest

from src.domain.ports import (
    BrokerPort,
    MarketDataPort,
    NotificationPort,
    PredictionResultRepository,
    StockFeatureRepository,
)
from src.infrastructure.in_memory import (
    InMemoryBrokerAdapter,
    InMemoryMarketDataAdapter,
    InMemoryNotificationAdapter,
    InMemoryPredictionRepository,
    InMemoryStockFeatureRepository,
)

# ---------------------------------------------------------------------------
# ポートが ABC として正しく定義されているか
# ---------------------------------------------------------------------------


def test_prediction_result_repository_is_abstract():
    with pytest.raises(TypeError):
        PredictionResultRepository()  # type: ignore[abstract]


def test_stock_feature_repository_is_abstract():
    with pytest.raises(TypeError):
        StockFeatureRepository()  # type: ignore[abstract]


def test_notification_port_is_abstract():
    with pytest.raises(TypeError):
        NotificationPort()  # type: ignore[abstract]


def test_market_data_port_is_abstract():
    with pytest.raises(TypeError):
        MarketDataPort()  # type: ignore[abstract]


def test_broker_port_is_abstract():
    with pytest.raises(TypeError):
        BrokerPort()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# InMemoryPredictionRepository
# ---------------------------------------------------------------------------


class TestInMemoryPredictionRepository:
    def _make_result(self, market: str, symbol: str):
        class _R:
            def __init__(self, market: str, symbol: str, diff_ratio: float) -> None:
                self.market = market
                self.symbol = symbol
                self.diff_ratio = diff_ratio

        return _R(market=market, symbol=symbol, diff_ratio=0.02)

    def test_save_and_load(self):
        repo = InMemoryPredictionRepository()
        result = self._make_result("us", "AAPL")
        repo.save("20260518_090000", [result])
        df = repo.load("us")
        assert not df.empty
        assert df.iloc[0]["market"] == "us"
        assert df.iloc[0]["symbol"] == "AAPL"

    def test_load_filters_by_market(self):
        repo = InMemoryPredictionRepository()
        repo.save("20260518_090000", [self._make_result("us", "AAPL")])
        repo.save("20260518_090000", [self._make_result("jp", "7203")])
        df = repo.load("us")
        assert all(df["market"] == "us")

    def test_load_empty_returns_empty_df(self):
        repo = InMemoryPredictionRepository()
        df = repo.load("us")
        assert df.empty

    def test_load_respects_limit(self):
        repo = InMemoryPredictionRepository()
        for i in range(5):
            r = self._make_result("us", f"SYM{i}")
            repo.save("20260518_090000", [r])
        df = repo.load("us", limit=3)
        assert len(df) == 3


# ---------------------------------------------------------------------------
# InMemoryStockFeatureRepository
# ---------------------------------------------------------------------------


class TestInMemoryStockFeatureRepository:
    def _make_df(self, rows: int = 3) -> pd.DataFrame:
        return pd.DataFrame({"close": [100.0 + i for i in range(rows)]})

    def test_upsert_and_load(self):
        repo = InMemoryStockFeatureRepository()
        df = self._make_df()
        repo.upsert("us", "AAPL", df)
        loaded = repo.load("us", "AAPL")
        assert loaded is not None
        assert len(loaded) == 3

    def test_load_returns_none_when_missing(self):
        repo = InMemoryStockFeatureRepository()
        assert repo.load("us", "AAPL") is None

    def test_upsert_overwrites(self):
        repo = InMemoryStockFeatureRepository()
        repo.upsert("us", "AAPL", self._make_df(3))
        repo.upsert("us", "AAPL", self._make_df(5))
        assert len(repo.load("us", "AAPL")) == 5

    def test_get_all_symbols(self):
        repo = InMemoryStockFeatureRepository()
        repo.upsert("us", "AAPL", self._make_df())
        repo.upsert("jp", "7203", self._make_df())
        symbols = repo.get_all_symbols()
        assert ("us", "AAPL") in symbols
        assert ("jp", "7203") in symbols

    def test_load_all_includes_market_symbol_columns(self):
        repo = InMemoryStockFeatureRepository()
        repo.upsert("us", "AAPL", self._make_df())
        df = repo.load_all()
        assert "market" in df.columns
        assert "symbol" in df.columns

    def test_load_all_empty_when_no_data(self):
        repo = InMemoryStockFeatureRepository()
        assert repo.load_all().empty


# ---------------------------------------------------------------------------
# InMemoryNotificationAdapter
# ---------------------------------------------------------------------------


class TestInMemoryNotificationAdapter:
    def test_send_message_succeeds(self):
        adapter = InMemoryNotificationAdapter()
        result = adapter.send_message("hello")
        assert result is True
        assert len(adapter.sent_messages) == 1
        assert adapter.sent_messages[0]["message"] == "hello"

    def test_send_message_fails_when_should_fail(self):
        adapter = InMemoryNotificationAdapter()
        adapter.should_fail = True
        result = adapter.send_message("hello")
        assert result is False
        assert len(adapter.sent_messages) == 0

    def test_send_message_stores_webhook_url(self):
        adapter = InMemoryNotificationAdapter()
        adapter.send_message("msg", webhook_url="https://example.com/webhook")
        assert adapter.sent_messages[0]["webhook_url"] == "https://example.com/webhook"


# ---------------------------------------------------------------------------
# InMemoryMarketDataAdapter
# ---------------------------------------------------------------------------


class TestInMemoryMarketDataAdapter:
    def test_get_stock_data_returns_set_data(self):
        adapter = InMemoryMarketDataAdapter()
        df = pd.DataFrame({"close": [100.0, 101.0]})
        adapter.set_stock_data("AAPL", "us", df)
        result = adapter.get_stock_data("AAPL", "us", "2026-01-01", "2026-01-31")
        assert not result.empty
        assert list(result["close"]) == [100.0, 101.0]

    def test_get_stock_data_returns_empty_when_not_set(self):
        adapter = InMemoryMarketDataAdapter()
        result = adapter.get_stock_data("AAPL", "us", "2026-01-01", "2026-01-31")
        assert result.empty

    def test_get_forex_data_returns_set_data(self):
        adapter = InMemoryMarketDataAdapter()
        df = pd.DataFrame({"close": [155.0]})
        adapter.set_forex_data("JPY=X", df)
        result = adapter.get_forex_data("JPY=X", "2026-01-01", "2026-01-31")
        assert not result.empty

    def test_get_ohlcv_returns_set_data(self):
        adapter = InMemoryMarketDataAdapter()
        df = pd.DataFrame({"Close": [100.0, 101.0], "Volume": [1000, 2000]})
        adapter.set_ohlcv_data("7203.T", df)
        result = adapter.get_ohlcv("7203.T", "5d")
        assert not result.empty
        assert list(result["Close"]) == [100.0, 101.0]

    def test_get_ohlcv_returns_empty_when_not_set(self):
        adapter = InMemoryMarketDataAdapter()
        result = adapter.get_ohlcv("AAPL", "5d")
        assert result.empty

    def test_get_latest_price_returns_set_price(self):
        adapter = InMemoryMarketDataAdapter()
        adapter.set_latest_price("AAPL", 150.0)
        assert adapter.get_latest_price("AAPL") == 150.0

    def test_get_latest_price_returns_zero_when_not_set(self):
        adapter = InMemoryMarketDataAdapter()
        assert adapter.get_latest_price("AAPL") == 0.0


# ---------------------------------------------------------------------------
# InMemoryBrokerAdapter
# ---------------------------------------------------------------------------


class TestInMemoryBrokerAdapter:
    def test_get_token(self):
        broker = InMemoryBrokerAdapter()
        assert broker.get_token() == "test_token"

    def test_get_balance(self):
        broker = InMemoryBrokerAdapter(initial_balance=500_000.0)
        assert broker.get_balance() == 500_000.0

    def test_send_order_returns_order_id(self):
        broker = InMemoryBrokerAdapter()
        result = broker.send_order("AAPL", side=1, qty=10)
        assert "order_id" in result
        assert result["status"] == "accepted"

    def test_get_orders_accumulates(self):
        broker = InMemoryBrokerAdapter()
        broker.send_order("AAPL", side=1, qty=10)
        broker.send_order("GOOG", side=1, qty=5)
        assert len(broker.get_orders()) == 2

    def test_cancel_order(self):
        broker = InMemoryBrokerAdapter()
        result = broker.cancel_order("order_1")
        assert result["status"] == "cancelled"

    def test_get_positions_initially_empty(self):
        broker = InMemoryBrokerAdapter()
        assert broker.get_positions() == []

    def test_get_short_positions_default_empty(self):
        broker = InMemoryBrokerAdapter()
        assert broker.get_short_positions() == []

    def test_implements_broker_port(self):
        broker = InMemoryBrokerAdapter()
        assert isinstance(broker, BrokerPort)
