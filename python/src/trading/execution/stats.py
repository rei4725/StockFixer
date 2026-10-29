"""OrderExecutionPipeline の集計型定義。"""

from typing import TypedDict


class OrderExecutionStats(TypedDict):
    buy_orders: int
    sell_orders: int
    short_orders: int
    skipped: int
    skipped_min_change: int
    errors: int
    trading_stopped: bool
    stop_reason: str | None
    reason_code: str | None
    daily_loss: float
    daily_loss_limit: float | None
    total_turnover: float
    correlation_blocked: bool
    enc: float
    avg_correlation: float
    n_held_symbols: int
    held_symbols_list: list[str]
    sl_tp_unevaluable_symbols: list[str]
    stale_prediction_reason: str | None


def _new_stats() -> OrderExecutionStats:
    """run_daily_orders の集計の初期値を返す。"""
    return {
        "buy_orders": 0,
        "sell_orders": 0,
        "short_orders": 0,
        "skipped": 0,
        "skipped_min_change": 0,
        "errors": 0,
        "trading_stopped": False,
        "stop_reason": None,
        "reason_code": None,
        "daily_loss": 0.0,
        "daily_loss_limit": None,
        "total_turnover": 0.0,
        "correlation_blocked": False,
        "enc": 0.0,
        "avg_correlation": 0.0,
        "n_held_symbols": 0,
        "held_symbols_list": [],
        "sl_tp_unevaluable_symbols": [],
        "stale_prediction_reason": None,
    }
