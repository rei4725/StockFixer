"""インメモリアダプター（テスト用）

各ポートのインメモリ実装。ユニットテストで DuckDB や外部 API なしに動作確認できる。
"""

from typing import Any, Optional

import pandas as pd

from src.domain.ports import (
    AlertLevel,
    BrokerPort,
    MarketDataPort,
    NotificationPort,
    PredictionResultRepository,
    StockFeatureRepository,
)


class InMemoryPredictionRepository(PredictionResultRepository):
    """インメモリ予測結果リポジトリ（テスト用）"""

    def __init__(self) -> None:
        self._records: list[dict] = []

    def save(self, predicted_at: str, results: list) -> None:
        for r in results:
            row = vars(r).copy() if hasattr(r, "__dict__") else dict(r)
            row["predicted_at"] = predicted_at
            self._records.append(row)

    def load(self, market: str, limit: int = 100) -> pd.DataFrame:
        rows = [r for r in self._records if r.get("market") == market]
        df = pd.DataFrame(rows)
        if not df.empty and limit:
            df = df.tail(limit)
        return df

    def get_latest_by_market(self, market: str) -> pd.DataFrame:
        rows = [r for r in self._records if r.get("market") == market]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if "predicted_at" in df.columns:
            idx = df.groupby("symbol")["predicted_at"].idxmax()
            df = df.loc[idx].reset_index(drop=True)
        if "diff_ratio" in df.columns:
            df = df[df["diff_ratio"].notna()].sort_values("diff_ratio", ascending=False)
        return df.reset_index(drop=True)


class InMemoryStockFeatureRepository(StockFeatureRepository):
    """インメモリ株式特徴量リポジトリ（テスト用）"""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], pd.DataFrame] = {}

    def upsert(self, market: str, symbol: str, df: pd.DataFrame) -> None:
        self._store[(market, symbol)] = df.copy()

    def load(self, market: str, symbol: str) -> Optional[pd.DataFrame]:
        return self._store.get((market, symbol))

    def load_all(self) -> pd.DataFrame:
        if not self._store:
            return pd.DataFrame()
        frames = []
        for (market, symbol), df in self._store.items():
            copy = df.copy()
            copy["market"] = market
            copy["symbol"] = symbol
            frames.append(copy)
        return pd.concat(frames, ignore_index=True)

    def get_all_symbols(self) -> list:
        return list(self._store.keys())


class InMemoryNotificationAdapter(NotificationPort):
    """インメモリ通知アダプター（テスト用）"""

    def __init__(self) -> None:
        self.sent_messages: list[dict] = []
        self.sent_alerts: list[dict] = []
        self.should_fail: bool = False

    def send_message(self, message: str, webhook_url: Optional[str] = None) -> bool:
        if self.should_fail:
            return False
        self.sent_messages.append({"message": message, "webhook_url": webhook_url})
        return True

    def send_alert(self, title: str, message: str, level: AlertLevel = AlertLevel.INFO) -> bool:
        if self.should_fail:
            return False
        self.sent_alerts.append({"title": title, "message": message, "level": level})
        return True


class NullNotificationAdapter(NotificationPort):
    """何も送信しない通知アダプター（テスト・無効化用）"""

    def send_message(self, message: str, webhook_url: Optional[str] = None) -> bool:
        return True

    def send_alert(self, title: str, message: str, level: AlertLevel = AlertLevel.INFO) -> bool:
        return True


class InMemoryMarketDataAdapter(MarketDataPort):
    """インメモリ市場データアダプター（テスト用）"""

    def __init__(self) -> None:
        self._stock_data: dict[tuple[str, str], pd.DataFrame] = {}
        self._forex_data: dict[str, pd.DataFrame] = {}
        self._ohlcv_data: dict[str, pd.DataFrame] = {}
        self._latest_prices: dict[str, float] = {}

    def set_stock_data(self, symbol: str, market: str, df: pd.DataFrame) -> None:
        self._stock_data[(symbol, market)] = df

    def set_forex_data(self, pair: str, df: pd.DataFrame) -> None:
        self._forex_data[pair] = df

    def set_ohlcv_data(self, symbol: str, df: pd.DataFrame) -> None:
        self._ohlcv_data[symbol] = df

    def set_latest_price(self, symbol: str, price: float) -> None:
        self._latest_prices[symbol] = price

    def get_stock_data(
        self,
        symbol: str,
        market: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        return self._stock_data.get((symbol, market), pd.DataFrame())

    def get_forex_data(
        self,
        pair: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        return self._forex_data.get(pair, pd.DataFrame())

    def get_ohlcv(self, symbol: str, period: str) -> pd.DataFrame:
        return self._ohlcv_data.get(symbol, pd.DataFrame())

    def get_latest_price(self, symbol: str) -> float:
        return self._latest_prices.get(symbol, 0.0)


class InMemoryBrokerAdapter(BrokerPort):
    """インメモリブローカーアダプター（テスト用）"""

    broker_name = "in_memory"

    def __init__(self, initial_balance: float = 1_000_000.0) -> None:
        self._balance = initial_balance
        self._positions: list[dict[str, Any]] = []
        self._orders: list[dict[str, Any]] = []

    def get_token(self) -> str:
        return "test_token"

    def send_order(
        self,
        symbol: str,
        side: int,
        qty: int,
        price: float = 0.0,
        order_type: int = 10,
    ) -> dict[str, Any]:
        order_id = f"order_{len(self._orders) + 1}"
        self._orders.append({"order_id": order_id, "symbol": symbol, "side": side, "qty": qty})
        return {"order_id": order_id, "status": "accepted", "message": ""}

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        return {"order_id": order_id, "status": "cancelled", "message": ""}

    def get_positions(self) -> list[dict[str, Any]]:
        return self._positions

    def get_balance(self) -> float:
        return self._balance

    def get_orders(self) -> list[dict[str, Any]]:
        return self._orders
