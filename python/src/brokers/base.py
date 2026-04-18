"""
BrokerBase — 証券会社連携の抽象基底クラス

各 Broker 実装（KabuBroker, PaperBroker 等）はこのクラスを継承する。
services 層は BrokerBase のみを参照し、具体的な証券会社に依存しない。
"""

from abc import ABC, abstractmethod
from enum import IntEnum


class OrderSide(IntEnum):
    BUY = 1
    SELL = 2  # ロング決済
    SHORT = 3  # 新規空売り（R-215）
    SHORT_COVER = 4  # 空売り返済（R-215）


class OrderType(IntEnum):
    MARKET = 10  # 成行
    LIMIT = 20  # 指値


class BrokerBase(ABC):
    """証券会社との注文・照会操作を抽象化するインターフェース"""

    # サブクラスで上書きする識別子（ログ・DB記録用）
    broker_name: str = "base"

    @abstractmethod
    def get_token(self) -> str:
        """認証トークンを取得・更新して返す"""

    @abstractmethod
    def send_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        price: float = 0.0,
        order_type: OrderType = OrderType.MARKET,
    ) -> dict:
        """
        注文を発注する。

        Args:
            symbol: 銘柄コード（例: "7203"）
            side: OrderSide.BUY or OrderSide.SELL
            qty: 株数
            price: 指値価格（成行の場合は 0.0）
            order_type: OrderType.MARKET or OrderType.LIMIT

        Returns:
            {"order_id": str, "status": str, "message": str}
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> dict:
        """注文をキャンセルする"""

    @abstractmethod
    def get_positions(self) -> list[dict]:
        """
        保有ポジション一覧を返す。

        Returns:
            [{"symbol": str, "qty": int, "avg_price": float, "current_price": float}, ...]
        """

    @abstractmethod
    def get_balance(self) -> float:
        """現金余力（円）を返す"""

    @abstractmethod
    def get_orders(self) -> list[dict]:
        """当日の注文一覧を返す"""
