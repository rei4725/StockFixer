"""長期バックテストの取引台帳。

役割は 3 つ。
  1. 監査証跡
  2. 「現金 + 保有評価 == equity」の不変条件をテストする根拠
  3. 銘柄・数量を取り違えない損益列の出所

metrics/core.compute_metrics には渡さない。core の FIFO 突合は銘柄も数量も
見ないため、多銘柄同時保有では損益が誤る（spec 参照）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

_COLUMNS = ["date", "action", "symbol", "price", "qty", "amount", "cash"]


@dataclass(frozen=True)
class TradeRecord:
    """1 件の約定。amount は符号なし金額、cash は約定後の現金残高。"""

    date: str
    action: str
    symbol: str
    price: float
    qty: float
    amount: float
    cash: float


class TradeLedger:
    """約定を発生順に追記する台帳。"""

    def __init__(self) -> None:
        self.records: List[TradeRecord] = []

    def record_buy(
        self, date: str, symbol: str, price: float, qty: float, amount: float, cash: float
    ) -> None:
        self.records.append(TradeRecord(date, "buy", symbol, price, qty, amount, cash))

    def record_sell(
        self,
        date: str,
        symbol: str,
        price: float,
        qty: float,
        amount: float,
        cash: float,
        action: str = "sell",
    ) -> None:
        self.records.append(TradeRecord(date, action, symbol, price, qty, amount, cash))

    def net_cash_flow(self) -> float:
        """買いの支払総額 − 売りの受取総額。現金の減少分と一致すべき値。"""
        paid = sum(r.amount for r in self.records if r.action == "buy")
        received = sum(r.amount for r in self.records if r.action != "buy")
        return float(paid - received)

    def to_frame(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=_COLUMNS)
        return pd.DataFrame([vars(r) for r in self.records], columns=_COLUMNS)
