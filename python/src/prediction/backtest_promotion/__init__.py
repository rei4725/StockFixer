"""バックテストによる仮の入れ替え（候補の勝ち判定・見張り）。"""

from src.prediction.backtest_promotion.judge import (
    HOLDOUT_TRADING_DAYS,
    MIN_COVERAGE,
    PATH_CHECK_SAMPLE,
    PATH_CHECK_TOL,
    ROLLBACK_HIT_MARGIN,
    UP_RATE_HIGH,
    UP_RATE_LOW,
    WATCH_TRADING_DAYS,
    judge_rollback,
    judge_win,
)

__all__ = [
    "HOLDOUT_TRADING_DAYS",
    "MIN_COVERAGE",
    "PATH_CHECK_SAMPLE",
    "PATH_CHECK_TOL",
    "ROLLBACK_HIT_MARGIN",
    "UP_RATE_HIGH",
    "UP_RATE_LOW",
    "WATCH_TRADING_DAYS",
    "judge_rollback",
    "judge_win",
]
