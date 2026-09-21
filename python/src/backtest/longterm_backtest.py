"""長期コホート・バックテストのファサード（互換維持用）。

実装は src/backtest/longterm/ に分割済み。本モジュールは CLI の import 互換の
ために残す。テストからのモジュール属性 patch は src.backtest.longterm.engine
を対象にすること（本モジュールへの patch は engine に届かない）。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.backtest.execution import DEFAULT_FEE_RATE
from src.backtest.longterm.config import LongtermBacktestConfig
from src.backtest.longterm.engine import enter_candidates  # noqa: F401
from src.backtest.longterm.engine import run_longterm_backtest as _run_with_config
from src.backtest.longterm.reporting import build_conclusion, save_results  # noqa: F401
from src.screening.types import HoldRules

__all__ = [
    "LongtermBacktestConfig",
    "build_conclusion",
    "run_longterm_backtest",
    "save_results",
]


def run_longterm_backtest(
    market: str = "us",
    start: str = "2021-01-01",
    end: str = "2026-01-01",
    rescreen_freq: str = "quarterly",
    top_n: int = 30,
    initial_cash: float = 1_000_000.0,
    max_positions: int = 10,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage: Optional[float] = None,
    benchmark_ticker: str = "^GSPC",
    execution_lag: int = 1,
    rules: Optional[HoldRules] = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """旧シグネチャの互換ラッパ。新規の呼び出しは Config 版を使うこと。"""
    config = LongtermBacktestConfig.build(
        market=market,
        start=start,
        end=end,
        rescreen_freq=rescreen_freq,
        top_n=top_n,
        initial_cash=initial_cash,
        max_positions=max_positions,
        fee_rate=fee_rate,
        slippage=slippage,
        benchmark_ticker=benchmark_ticker,
        execution_lag=execution_lag,
        rules=rules,
    )
    return _run_with_config(config)
