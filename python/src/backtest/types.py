"""
backtest BC の型定義。

バックテスト・ストレステスト・Walk-Forward で使用するドメイン型。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StressTestResult:
    """ストレステスト1銘柄・1シナリオの結果。"""

    market: str
    symbol: str
    scenario_name: str  # "corona" / "lehman"
    period_start: str  # "2020-02-01"
    period_end: str  # "2020-03-31"
    mdd: float  # 最大ドローダウン（負の値）
    sharpe_ratio: float
    total_return: float
    win_rate: float
    num_trades: int
    max_consecutive_losses: int
    mdd_pass: bool  # abs(mdd) <= MDD_THRESHOLD (0.15)
