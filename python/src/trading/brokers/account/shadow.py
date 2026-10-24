"""
ShadowBroker — 段3の影の運用。紙（PaperBroker）を正とし、同じ買い・売りを口座アプリにも出す

口座アプリの答えで本体の動きは変えない（写しの失敗はのみこむ）。どちらに何を出したかは
results/account_shadow/YYYY-MM-DD.jsonl に1注文1行で残し、17:00 の日報（shadow_report）が読む。
"""

import json
import os
from collections.abc import Callable
from datetime import date
from typing import Any

from src.trading.brokers.account.gateway import AccountGatewayBroker
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.utils.data_path_utils import get_results_dir
from src.utils.japan_time import isoformat_jst, now_jst
from src.utils.logger import get_logger

logger = get_logger(__name__)

_FORWARDED = (OrderSide.BUY, OrderSide.SELL)


def shadow_log_dir() -> str:
    return os.path.join(get_results_dir(), "account_shadow")


class ShadowBroker(BrokerBase):
    """読むのは全部紙から。注文は紙に出した後、買い・売りだけ口座アプリにも出す"""

    def __init__(
        self,
        primary: BrokerBase,
        shadow: AccountGatewayBroker,
        *,
        job: str,
        log_dir: str | None = None,
        today: Callable[[], date] | None = None,
    ) -> None:
        self._primary = primary
        self._shadow = shadow
        self._job = job
        self._log_dir = log_dir or shadow_log_dir()
        self._today = today or (lambda: now_jst().date())
        # 本体の分かれ道（空売りは paper だけ、など）を変えない
        self.broker_name = primary.broker_name

    def __getattr__(self, name: str) -> Any:
        # settle_pending_orders・get_pnl_summary など紙だけの道は紙に渡す
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._primary, name)

    def get_token(self) -> str:
        return self._primary.get_token()

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        return self._primary.cancel_order(order_id)

    def get_positions(self) -> list[dict[str, Any]]:
        return self._primary.get_positions()

    def get_balance(self) -> float:
        return self._primary.get_balance()

    def get_orders(self) -> list[dict[str, Any]]:
        return self._primary.get_orders()

    def get_short_positions(self) -> list[dict[str, Any]]:
        return self._primary.get_short_positions()

    def send_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        price: float = 0.0,
        order_type: OrderType = OrderType.MARKET,
    ) -> dict[str, Any]:
        result = self._primary.send_order(symbol, side, qty, price=price, order_type=order_type)
        record: dict[str, Any] = {
            "at": isoformat_jst(),
            "job": self._job,
            "symbol": symbol.replace(".T", ""),
            "side": side.name.lower(),
            "qty": int(qty),
            "order_type": order_type.name.lower(),
            "price": float(price),
            "paper": result,
        }
        if side in _FORWARDED:
            try:
                record["account"] = self._shadow.send_order(
                    symbol, side, qty, price=price, order_type=order_type
                )
            except Exception as e:
                logger.warning(
                    "[shadow] 口座アプリへの写しが失敗（紙の注文はそのまま）: %r", e, exc_info=True
                )
                record["account"] = {"status": "error", "message": repr(e)[:300]}
        else:
            record["account"] = {
                "status": "not_forwarded",
                "message": "空売りは口座アプリに写さない",
            }
        self._append(record)
        return result

    def _append(self, record: dict[str, Any]) -> None:
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            path = os.path.join(self._log_dir, f"{self._today().isoformat()}.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.error("[shadow] 写しの記録に失敗（紙の注文はそのまま）", exc_info=True)
