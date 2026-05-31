"""
KabuBroker — au カブコム証券 kabu STATION® API クライアント

kabu STATION® アプリ（常駐）が起動している前提で localhost:18080 を叩く。
APIパスワードは環境変数 KABU_API_PASSWORD から取得する。

公式仕様書: https://kabucom.github.io/kabusapi/reference/
"""

import os
from datetime import datetime, timedelta
from typing import Any

import httpx

from src.trading.brokers.base import BrokerBase, BrokerError, OrderSide, OrderType
from src.utils.logger import get_logger

logger = get_logger(__name__)

_BASE_URL = "http://localhost:18080/kabusapi"
_TOKEN_TTL_SECONDS = 3600  # トークン有効期間（1時間）
_TOKEN_REFRESH_BUFFER_SECONDS = 300  # 期限5分前に事前更新
_TOKEN_MAX_RETRIES = 2  # 1回リトライ（初回 + 1回）


class KabuBroker(BrokerBase):
    """kabu STATION® API を使った証券操作クライアント"""

    broker_name = "kabu"

    def __init__(self, api_password: str | None = None, timeout: float = 10.0):
        """
        Args:
            api_password: kabu STATION® API パスワード。
                          省略時は環境変数 KABU_API_PASSWORD を使用。
            timeout: HTTP タイムアウト（秒）
        """
        self._api_password = api_password or os.environ.get("KABU_API_PASSWORD", "")
        self._timeout = timeout
        self._token: str = ""
        self._token_expires_at: datetime = datetime.min

    # ------------------------------------------------------------------
    # 認証
    # ------------------------------------------------------------------

    def get_token(self) -> str:
        """
        APIトークンを取得する。
        キャッシュ済みかつ期限まで5分超あれば再利用する。
        取得失敗時は最大1回リトライし、それでも失敗すれば BrokerError を raise する。
        """
        buffer = timedelta(seconds=_TOKEN_REFRESH_BUFFER_SECONDS)
        if self._token and datetime.now() < self._token_expires_at - buffer:
            return self._token

        if not self._api_password:
            raise EnvironmentError(
                "KABU_API_PASSWORD が設定されていません。" "環境変数を設定するか KabuBroker(api_password=...) で渡してください。"
            )

        last_exc: Exception | None = None
        for attempt in range(_TOKEN_MAX_RETRIES):
            try:
                resp = httpx.post(
                    f"{_BASE_URL}/token",
                    json={"APIPassword": self._api_password},
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                self._token = resp.json()["Token"]
                self._token_expires_at = datetime.now() + timedelta(seconds=_TOKEN_TTL_SECONDS)
                logger.info("kabu STATION® API トークンを取得しました (attempt=%d)", attempt + 1)
                return self._token
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "kabu STATION® API トークン取得失敗 (attempt=%d/%d): %s",
                    attempt + 1,
                    _TOKEN_MAX_RETRIES,
                    exc,
                )

        raise BrokerError(
            f"kabu STATION® API トークン取得に失敗しました ({_TOKEN_MAX_RETRIES}回試行): {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # 注文
    # ------------------------------------------------------------------

    def send_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        price: float = 0.0,
        order_type: OrderType = OrderType.MARKET,
    ) -> dict[str, Any]:
        """
        現物株式の注文を発注する（東証・現物・特定口座）。

        Args:
            symbol: 銘柄コード（".T" なし、例: "7203"）
            side: OrderSide.BUY / SELL
            qty: 株数
            price: 指値価格（成行は 0.0）
            order_type: OrderType.MARKET / LIMIT

        Returns:
            {"order_id": str, "status": str, "message": str}
        """
        token = self.get_token()
        body = {
            "Password": os.environ.get("KABU_TRADE_PASSWORD", ""),
            "Symbol": symbol.replace(".T", ""),
            "Exchange": 1,  # 1 = 東証
            "SecurityType": 1,  # 1 = 株式
            "Side": str(int(side)),
            "CashMargin": 1,  # 1 = 現物
            "DelivType": 2,  # 2 = お預り金
            "FundType": "AA",  # AA = 特定預り
            "Qty": qty,
            "FrontOrderType": int(order_type),
            "Price": price if order_type == OrderType.LIMIT else 0,
            "ExpireDay": 0,  # 0 = 当日有効
        }
        logger.info(f"[kabu] 注文送信: symbol={symbol} side={side.name} qty={qty} price={price}")
        resp = httpx.post(
            f"{_BASE_URL}/sendorder",
            headers={"X-API-KEY": token},
            json=body,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        order_id = data.get("OrderId", "")
        logger.info(f"[kabu] 注文受付: order_id={order_id}")
        return {"order_id": order_id, "status": "accepted", "message": str(data)}

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """注文をキャンセルする"""
        token = self.get_token()
        body = {
            "OrderId": order_id,
            "Password": os.environ.get("KABU_TRADE_PASSWORD", ""),
        }
        resp = httpx.put(
            f"{_BASE_URL}/cancelorder",
            headers={"X-API-KEY": token},
            json=body,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        logger.info(f"[kabu] 注文キャンセル: order_id={order_id}")
        return {"order_id": order_id, "status": "cancelled", "message": str(resp.json())}

    # ------------------------------------------------------------------
    # 照会
    # ------------------------------------------------------------------

    def get_positions(self) -> list[dict[str, Any]]:
        """保有ポジション一覧を返す"""
        token = self.get_token()
        resp = httpx.get(
            f"{_BASE_URL}/positions",
            headers={"X-API-KEY": token},
            params={"product": 0},  # 0 = 全商品
            timeout=self._timeout,
        )
        resp.raise_for_status()
        raw = resp.json() or []
        return [
            {
                "symbol": item.get("Symbol", ""),
                "qty": item.get("LeavesQty", 0),
                "avg_price": item.get("Price", 0.0),
                "current_price": item.get("CurrentPrice", 0.0),
            }
            for item in raw
        ]

    def get_balance(self) -> float:
        """現金余力（円）を返す"""
        token = self.get_token()
        resp = httpx.get(
            f"{_BASE_URL}/wallet/cash",
            headers={"X-API-KEY": token},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return float(resp.json().get("StockAccountWallet", 0.0))

    def get_orders(self) -> list[dict[str, Any]]:
        """当日の注文一覧を返す"""
        token = self.get_token()
        resp = httpx.get(
            f"{_BASE_URL}/orders",
            headers={"X-API-KEY": token},
            params={"product": 0, "details": True},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        raw = resp.json() or []
        return [
            {
                "order_id": item.get("ID", ""),
                "symbol": item.get("Symbol", ""),
                "side": item.get("Side", ""),
                "qty": item.get("OrderQty", 0),
                "price": item.get("Price", 0.0),
                "status": item.get("State", ""),
            }
            for item in raw
        ]
