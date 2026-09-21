"""長期バックテストの保有ポジションとポートフォリオ集約。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from src.backtest.execution import ExecutionModel
from src.backtest.longterm.ledger import TradeLedger
from src.screening.types import PositionEvent


@dataclass
class OpenPosition:
    """保有中の 1 銘柄。

    cost_basis は手数料・スリッページ込みの取得原価で、損益計算の正本である。
    realized は部分利確で既に受け取った金額の累計。
    """

    symbol: str
    entry_date: str
    entry_price: float
    shares: int
    current_hf: float = 1.0
    cost_basis: float = 0.0
    realized: float = 0.0
    events_by_date: Dict[str, List[PositionEvent]] = field(default_factory=dict)

    def held_shares(self) -> float:
        return self.shares * self.current_hf

    def market_value(self, price: float) -> float:
        return self.held_shares() * price


@dataclass(frozen=True)
class ClosedTrade:
    """決済済みポジション 1 件。trades_df の 1 行 + 損益。"""

    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    multiple: float
    max_multiple: float
    exit_reason: str
    held_days: int
    realized_pnl: float
    return_rate: float


class Portfolio:
    """現金と保有ポジションを持つ可変の集約。

    日次ループで 1200 回以上更新されるため frozen にはしない。
    """

    def __init__(self, cash: float) -> None:
        self.cash = cash
        self.positions: Dict[str, OpenPosition] = {}
        self.ledger = TradeLedger()
        self.closed: List[ClosedTrade] = []

    def enter(self, pos: OpenPosition) -> None:
        """取得原価 pos.cost_basis を現金から引いて保有に加える。"""
        self.cash -= pos.cost_basis
        self.positions[pos.symbol] = pos
        self.ledger.record_buy(
            pos.entry_date, pos.symbol, pos.entry_price, pos.shares, pos.cost_basis, self.cash
        )

    def scale_out(self, symbol: str, ev: PositionEvent, execution: ExecutionModel) -> None:
        """部分利確。保有比率の減少分だけ売却して現金を回収する。"""
        pos = self.positions[symbol]
        delta = pos.current_hf - ev.held_fraction
        if delta <= 0:
            return
        sold = pos.shares * delta
        proceeds = execution.sell_proceeds(sold, ev.price)
        self.cash += proceeds
        pos.realized += proceeds
        pos.current_hf = ev.held_fraction
        self.ledger.record_sell(
            ev.date, symbol, ev.price, sold, proceeds, self.cash, action="scale_out"
        )

    def close(
        self,
        symbol: str,
        ev: PositionEvent,
        execution: ExecutionModel,
        max_multiple: float,
    ) -> ClosedTrade:
        """残りを全売却してポジションを閉じ、確定した ClosedTrade を返す。

        max_multiple はエンジン側で追跡している「保有中の最大倍率」であり、
        既定値は与えない: 省略を許すとエンジンが渡し忘れたまま
        max_multiple 列が静かに壊れうるため、必須引数のまま維持する。
        """
        pos = self.positions.pop(symbol)
        sold = pos.held_shares()
        proceeds = execution.sell_proceeds(sold, ev.price)
        self.cash += proceeds
        pos.realized += proceeds
        pos.current_hf = 0.0
        self.ledger.record_sell(ev.date, symbol, ev.price, sold, proceeds, self.cash)

        pnl = pos.realized - pos.cost_basis
        rate = pnl / pos.cost_basis if pos.cost_basis > 0 else 0.0
        mult = ev.price / pos.entry_price if pos.entry_price > 0 else 0.0
        held_days = (pd.Timestamp(ev.date) - pd.Timestamp(pos.entry_date)).days

        trade = ClosedTrade(
            symbol=symbol,
            entry_date=pos.entry_date,
            exit_date=ev.date,
            entry_price=pos.entry_price,
            exit_price=ev.price,
            multiple=mult,
            max_multiple=max_multiple,
            exit_reason=ev.reason,
            held_days=held_days,
            realized_pnl=pnl,
            return_rate=rate,
        )
        self.closed.append(trade)
        return trade

    def equity(self, close_lookups: Dict[str, Dict[str, float]], date: str) -> float:
        """現金 + 保有の時価評価。"""
        held = sum(
            p.market_value(close_lookups.get(p.symbol, {}).get(date, 0.0))
            for p in self.positions.values()
        )
        return self.cash + held
