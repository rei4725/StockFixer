"""長期コホート・バックテストの設定値。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.backtest.execution import DEFAULT_FEE_RATE, TradingCosts
from src.screening.types import HoldRules


@dataclass(frozen=True)
class LongtermBacktestConfig:
    """1 回の長期バックテストを定義する不変の設定。

    execution_lag: エントリー約定をリスクリーン日から何営業日ずらすかを表す想定の
                   フィールド。**未配線（現時点ではどの値を渡しても効果が無い）**。
                   PR-5 でエンジンに配線される予定。
    n_trials:      DSR（過学習ガード）の試行回数を表す想定のフィールド。
                   **未配線（現時点ではどの値を渡しても効果が無い）**。
                   PR-6 で配線される予定。
    """

    market: str = "us"
    start: str = "2021-01-01"
    end: str = "2026-01-01"
    rescreen_freq: str = "quarterly"
    top_n: int = 30
    initial_cash: float = 1_000_000.0
    max_positions: int = 10
    execution_lag: int = 0
    benchmark_ticker: str = "^GSPC"
    n_trials: int = 0
    costs: TradingCosts = field(default_factory=TradingCosts)
    rules: HoldRules = field(default_factory=HoldRules)

    @classmethod
    def build(
        cls,
        *,
        market: str = "us",
        fee_rate: float = DEFAULT_FEE_RATE,
        slippage: Optional[float] = None,
        rules: Optional[HoldRules] = None,
        **kwargs: Any,
    ) -> "LongtermBacktestConfig":
        """市場別スリッページ既定の解決を含む組み立て。CLI はこれを呼ぶ。"""
        return cls(
            market=market,
            costs=TradingCosts.for_market(market, fee_rate, slippage),
            rules=rules or HoldRules(),
            **kwargs,
        )
