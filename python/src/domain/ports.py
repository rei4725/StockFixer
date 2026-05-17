"""ヘキサゴナルアーキテクチャ ポート定義

ドメイン層がインフラ層の実装詳細を知らずに依存できるインターフェース群。
アダプター実装は src/infrastructure/ に置く。
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd


class BrokerPort(ABC):
    """証券会社との注文・照会操作を抽象化するポート"""

    broker_name: str = "base"

    @abstractmethod
    def get_token(self) -> str:
        """認証トークンを取得・更新して返す"""

    @abstractmethod
    def send_order(
        self,
        symbol: str,
        side: int,
        qty: int,
        price: float = 0.0,
        order_type: int = 10,
    ) -> dict[str, Any]:
        """注文を発注する"""

    @abstractmethod
    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """注文をキャンセルする"""

    @abstractmethod
    def get_positions(self) -> list[dict[str, Any]]:
        """保有ポジション一覧を返す"""

    @abstractmethod
    def get_balance(self) -> float:
        """現金余力（円）を返す"""

    @abstractmethod
    def get_orders(self) -> list[dict[str, Any]]:
        """当日の注文一覧を返す"""

    def get_short_positions(self) -> list[dict[str, Any]]:
        return []


class PredictionResultRepository(ABC):
    """予測結果の永続化ポート"""

    @abstractmethod
    def save(self, predicted_at: str, results: list) -> None:
        """予測結果を保存する"""

    @abstractmethod
    def load(self, market: str, limit: int = 100) -> pd.DataFrame:
        """指定マーケットの予測結果を読み込む"""


class StockFeatureRepository(ABC):
    """株式特徴量の永続化ポート"""

    @abstractmethod
    def upsert(self, market: str, symbol: str, df: pd.DataFrame) -> None:
        """特徴量データを保存する（べき等）"""

    @abstractmethod
    def load(self, market: str, symbol: str) -> Optional[pd.DataFrame]:
        """1銘柄分の特徴量を読み込む"""

    @abstractmethod
    def load_all(self) -> pd.DataFrame:
        """全銘柄の特徴量を読み込む"""

    @abstractmethod
    def get_all_symbols(self) -> list:
        """全銘柄の (market, symbol) タプルリストを返す"""


class NotificationPort(ABC):
    """外部通知サービスのポート"""

    @abstractmethod
    def send_message(self, message: str, webhook_url: Optional[str] = None) -> bool:
        """メッセージを送信する。成功時 True を返す。"""


class MarketDataPort(ABC):
    """市場データ取得のポート"""

    @abstractmethod
    def get_stock_data(
        self,
        symbol: str,
        market: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """株価 OHLCV データを取得する"""

    @abstractmethod
    def get_forex_data(
        self,
        pair: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """為替レートデータを取得する"""
