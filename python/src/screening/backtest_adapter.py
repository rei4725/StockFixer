"""
BacktestScreeningAdapter - BacktestScreeningPort の具象実装。

screening BC の純粋関数を BacktestScreeningPort インターフェースに適合させる。
screening BC 内に配置することで backtest -> screening 直接依存を解消する。
Protocol は構造的なので、本モジュールは backtest を import しない。
"""

from __future__ import annotations

import pandas as pd

from src.domain.types import HoldRules, PositionEvent, TrendCandidate


class BacktestScreeningAdapter:
    """screening BC の関数を BacktestScreeningPort インターフェースに適合させるアダプター。"""

    def screen_trend_candidates(self, market: str, top_n: int, as_of: str) -> list[TrendCandidate]:
        from src.screening.trend_screener import screen_trend_candidates

        return screen_trend_candidates(market=market, top_n=top_n, as_of=as_of)

    def simulate_position(
        self, prices: pd.DataFrame, entry_date: str, rules: HoldRules
    ) -> list[PositionEvent]:
        from src.screening.hold_engine import simulate_position

        return simulate_position(prices, entry_date=entry_date, rules=rules)
