"""
BacktestScreeningPort - screening BC への依存を逆転させるポート定義。

backtest BC はこのポートを通じて screening BC の純粋関数を利用する。
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import pandas as pd

from src.domain.types import HoldRules, PositionEvent, TrendCandidate


@runtime_checkable
class BacktestScreeningPort(Protocol):
    """バックテスト用 screening アクセスポートのインターフェース。"""

    def screen_trend_candidates(
        self, market: str, top_n: int, as_of: str
    ) -> list[TrendCandidate]: ...

    def simulate_position(
        self, prices: pd.DataFrame, entry_date: str, rules: HoldRules
    ) -> list[PositionEvent]: ...


_port: Optional[BacktestScreeningPort] = None


def set_backtest_screening_port(port: BacktestScreeningPort) -> None:
    """テストや orchestration から実装を注入するためのセッター。"""
    global _port
    _port = port


def get_backtest_screening_port() -> BacktestScreeningPort:
    """注入済みの BacktestScreeningPort を返す。未注入なら RuntimeError。

    注入は orchestration の合成ルートで行う:
    `src.orchestration.port_wiring.wire_ports()`（エントリポイント起動時に呼ぶ）。
    テストや個別注入は `set_backtest_screening_port()` を使う。
    """
    if _port is None:
        raise RuntimeError(
            "BacktestScreeningPort が未注入です。エントリポイントで "
            "src.orchestration.port_wiring.wire_ports() を呼ぶか、"
            "set_backtest_screening_port() で実装を注入してください。"
        )
    return _port
