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
