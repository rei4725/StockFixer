"""取引コストと約定金額の計算を集約する（Phase 1 / PR-1）。

backtest BC には手数料・スリッページの演算が 21 箇所へ手書きで散っており、
次の不整合を生んでいた。

1. スリッページが `Backtester` にしか届かず、`portfolio/simulation.py` と
   `longterm_backtest.py` は手数料のみだった。
2. 手数料の掛け方の式が揃っていない（予算控除型と価格上乗せ型が混在）。
3. 既定値が呼び出し側ごとに二重管理されていた。

この module がコスト計算の唯一の正本である。市場別の既定スリッページ（#494）と
動的スリッページ（R-210 平方根インパクトモデル）の双方をここで合成する。

依存の向き:
    `config.settings` のみを参照し、`pipeline/runner.py` は参照しない。
    逆向きにすると runner が ExecutionModel を組むために本 module を import
    するため循環参照になる。

設計: docs/superpowers/specs/2026-09-21-execution-model-design.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

from config.settings import DEFAULT_SLIPPAGE_JP, DEFAULT_SLIPPAGE_US

# 手数料の既定率。CLI / エンジン双方の既定をここに一本化する。
DEFAULT_FEE_RATE: float = 0.001


def default_slippage_for(market: str) -> float:
    """市場別のデフォルト片道スリッページ率を返す（#494）。

    未知の market は US 既定にフォールバックする。
    """
    return DEFAULT_SLIPPAGE_JP if market == "jp" else DEFAULT_SLIPPAGE_US


@dataclass(frozen=True)
class TradingCosts:
    """片道の取引コスト率。

    Attributes:
        fee_rate: 売買手数料率（片道）
        slippage_rate: フラットな片道スリッページ率。サイズ・流動性に依存する
            上乗せ分は ExecutionModel 側の slippage_fn が担う。
    """

    fee_rate: float = 0.0
    slippage_rate: float = 0.0

    @classmethod
    def for_market(
        cls,
        market: str,
        fee_rate: float = DEFAULT_FEE_RATE,
        slippage: Optional[float] = None,
    ) -> "TradingCosts":
        """市場別の既定値からコストを組み立てる。

        Args:
            market: マーケット識別子（"us" / "jp" 等）
            fee_rate: 売買手数料率
            slippage: 片道スリッページ率。None のときのみ市場別既定を使う。
                0.0 は「コスト無し」の明示指定として尊重する。
        """
        resolved = default_slippage_for(market) if slippage is None else slippage
        return cls(fee_rate=fee_rate, slippage_rate=resolved)


class ExecutionModel:
    """コスト率から約定金額を求める。

    `slippage_fn` は R-210 の動的スリッページ関数 (qty, price, avg_volume) -> rate。
    与えられた場合、フラット率に市場インパクトを加算する。
    """

    def __init__(
        self,
        costs: TradingCosts,
        slippage_fn: Optional[Callable[[int, float, float], float]] = None,
    ) -> None:
        self.costs = costs
        self.slippage_fn = slippage_fn

    def effective_slippage(self, qty: float, price: float, volume: float = 0.0) -> float:
        """有効スリッページ率を返す（#494）。

        基準スプレッド（フラット率）に、slippage_fn が設定されていればサイズ・
        流動性依存のマーケットインパクト（R-210 平方根モデル）を加算する。
        出来高不明（volume<=0）や数量0の場合はフラット基準のみ。
        """
        base = self.costs.slippage_rate
        if self.slippage_fn is not None and volume > 0 and qty > 0:
            return base + self.slippage_fn(int(qty), price, volume)
        return base

    def unit_buy_cost(self, price: float, volume: float = 0.0, qty: int = 0) -> float:
        """1 株あたり取得原価 price * (1 + fee + slippage) を返す。

        qty を与えると動的スリッページが反映される（省略時はフラット率のみ）。
        """
        return price * (1.0 + self.costs.fee_rate + self.effective_slippage(qty, price, volume))

    def buy_cost(self, qty: float, price: float, volume: float = 0.0) -> float:
        """qty 株を買うのに必要な現金（手数料・スリッページ込み）を返す。"""
        slip = self.effective_slippage(qty, price, volume)
        return qty * price * (1.0 + self.costs.fee_rate + slip)

    def sell_proceeds(self, qty: float, price: float, volume: float = 0.0) -> float:
        """qty 株を売って得られる現金（手数料・スリッページ控除後）を返す。

        qty が float なのは、長期バックテストの部分利確が端株を売るため。
        """
        slip = self.effective_slippage(qty, price, volume)
        return qty * price * (1.0 - self.costs.fee_rate - slip)

    def max_affordable_qty(self, cash: float, price: float, volume: float = 0.0) -> int:
        """cash で買える最大株数を返す。

        買った後に現金が負にならないことを保証する（過剰買付 #483 の再発防止）。
        動的スリッページは数量に依存するため、暫定数量で見積もったコストを
        用いて算出し、超過していた場合のみ 1 株ずつ切り下げる。
        """
        if cash <= 0 or price <= 0:
            return 0

        qty = int(math.floor(cash / self.unit_buy_cost(price, volume)))
        if qty <= 0:
            return 0

        # 動的スリッページは数量が増えるほど不利になるため、見積もりが
        # 楽観側へ振れうる。実コストで検算して超過分を切り下げる。
        while qty > 0 and self.buy_cost(qty, price, volume) > cash:
            qty -= 1
        return qty
