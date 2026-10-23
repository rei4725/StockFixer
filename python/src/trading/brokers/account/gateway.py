"""
AccountGatewayBroker — 口座アプリ（rei4725/StockFixer-account）を HTTP で呼ぶ Broker

約束の正本は口座アプリの contracts/account-api.json で、本体は写しを repo の contracts/ に置く。
口座アプリが応えない・時間切れ・想定外の答えのときは、その日（JST）の残りの注文を出さない。
PaperBroker や KabuBroker に戻らない（危ない方に倒さない）。止めの印は state_dir に日ごとのファイルで残し、
翌日は自動で動き直す。同じ日に動き直すときは、人間がその日のファイルを消す。
"""

import json
import os
from collections.abc import Callable
from datetime import date
from typing import Any

import httpx

from src.domain.ports import AlertLevel, NotificationPort
from src.trading.brokers.base import BrokerBase, BrokerError, OrderSide, OrderType
from src.utils.data_path_utils import get_results_dir
from src.utils.japan_time import isoformat_jst, now_jst
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SIDES = {OrderSide.BUY: "buy", OrderSide.SELL: "sell"}
_TYPES = {OrderType.MARKET: "market", OrderType.LIMIT: "limit"}


class AccountGatewayUnavailable(BrokerError):
    """口座アプリが応えない・時間切れ・想定外の答え。その日の残りの注文は出さない"""


class AccountGatewayBroker(BrokerBase):
    """口座アプリの API（POST /v1/orders・GET /v1/positions・/v1/balance・/v1/orders）を呼ぶ"""

    broker_name = "account"

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        job: str,
        market: str = "jp",
        timeout: float = 10.0,
        state_dir: str | None = None,
        notifier: NotificationPort | None = None,
        transport: httpx.BaseTransport | None = None,
        today: Callable[[], date] | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
            transport=transport,
        )
        self._job = job
        self._market = market
        self._state_dir = state_dir or os.path.join(get_results_dir(), "account_gateway")
        self._notifier = notifier
        self._today = today or (lambda: now_jst().date())

    # ------------------------------------------------------------------
    # 止めの印
    # ------------------------------------------------------------------

    def _stop_path(self) -> str:
        return os.path.join(self._state_dir, f"stopped-{self._today().isoformat()}.json")

    def is_stopped_today(self) -> bool:
        return os.path.exists(self._stop_path())

    def _stop(self, what: str, err: Exception) -> AccountGatewayUnavailable:
        os.makedirs(self._state_dir, exist_ok=True)
        with open(self._stop_path(), "w", encoding="utf-8") as f:
            json.dump(
                {"job": self._job, "what": what, "error": repr(err)[:300], "at": isoformat_jst()},
                f,
                ensure_ascii=False,
            )
        logger.error(
            "[account] 口座アプリに届かないため、今日の残りの注文を出さない: %s %r", what, err
        )
        if self._notifier is not None:
            try:
                self._notifier.send_alert(
                    "口座アプリに届かない",
                    f"{what} が失敗したため、今日（JST）の口座アプリへの残りの注文を出しません。\n{err!r}",
                    level=AlertLevel.ERROR,
                )
            except Exception:
                logger.error("[account] 知らせの送信に失敗", exc_info=True)
        return AccountGatewayUnavailable(f"{what}: {err!r}")

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as e:
            raise self._stop(f"{method} {path}", e) from e

    # ------------------------------------------------------------------
    # 注文
    # ------------------------------------------------------------------

    def client_order_id(self, symbol: str, side: OrderSide) -> str:
        """日付（JST）-ジョブ-銘柄-売買。やり直しても同じ値になり、口座アプリが二重の注文を断つ"""
        return f"{self._today():%Y%m%d}-{self._job}-{symbol.replace('.T', '')}-{_SIDES[side]}"

    def get_token(self) -> str:
        """トークンは固定。届くかどうかを /health で確かめる"""
        self._call("GET", "/health")
        return "account"

    def send_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        price: float = 0.0,
        order_type: OrderType = OrderType.MARKET,
    ) -> dict[str, Any]:
        if side not in _SIDES:
            raise BrokerError(f"口座アプリは {side.name} を受けない（買いと売りだけ）")
        if self.is_stopped_today():
            raise AccountGatewayUnavailable("今日は口座アプリへの注文を止めている")
        sym = symbol.replace(".T", "")
        body: dict[str, Any] = {
            "client_order_id": self.client_order_id(sym, side),
            "symbol": sym,
            "market": self._market,
            "side": _SIDES[side],
            "qty": int(qty),
            "order_type": _TYPES[order_type],
            "reason": f"stockfixer:{self._job}",
        }
        if order_type == OrderType.LIMIT:
            body["limit_price"] = float(price)
        reply = self._call("POST", "/v1/orders", json=body)
        status = reply.get("status") if isinstance(reply, dict) else None
        if status == "accepted":
            logger.info(
                "[account] 受付: %s order_id=%s", body["client_order_id"], reply.get("order_id")
            )
            return {"order_id": str(reply["order_id"]), "status": "pending", "message": "accepted"}
        if status == "rejected":
            code = reply.get("reason_code") or "internal_error"
            logger.warning(
                "[account] 断り: %s %s %s", body["client_order_id"], code, reply.get("detail")
            )
            return {
                "order_id": None,
                "status": "rejected",
                "reason_code": code,
                "message": f"{code}: {reply.get('detail') or ''}",
            }
        raise self._stop("POST /v1/orders", ValueError(f"想定外の答え: {reply!r}"[:300]))

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        raise BrokerError("口座アプリは注文の取り消しを受けない")

    # ------------------------------------------------------------------
    # 照会
    # ------------------------------------------------------------------

    def get_positions(self) -> list[dict[str, Any]]:
        return [
            {
                "symbol": p["symbol"],
                "market": p["market"],
                "qty": int(p["qty"]),
                "avg_price": float(p["avg_price"]),
                "current_price": float(p["last_price"]),
            }
            for p in self._call("GET", "/v1/positions")
            if p["market"] == self._market
        ]

    def get_balance(self) -> float:
        return float(self._call("GET", "/v1/balance")["cash"])

    def get_equity(self) -> float:
        return float(self._call("GET", "/v1/balance")["equity"])

    def get_orders_on(self, day: date) -> list[dict[str, Any]]:
        return list(self._call("GET", "/v1/orders", params={"date": day.isoformat()}))

    def get_orders(self) -> list[dict[str, Any]]:
        return [
            {
                "order_id": str(o["id"]),
                "symbol": o["symbol"],
                "side": o["side"],
                "qty": int(o["qty"]),
                "price": o.get("limit_price"),
                "status": o["status"],
            }
            for o in self.get_orders_on(self._today())
        ]
