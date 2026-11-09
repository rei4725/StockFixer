"""OrderExecutionPipeline の集計型定義。"""

from typing import TypedDict


class OrderExecutionStats(TypedDict):
    buy_orders: int
    sell_orders: int
    short_orders: int
    skipped: int
    skipped_min_change: int
    skip_reasons: dict[str, int]  # skipped の理由ごとの数。和は skipped と一致する（#1765）
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
    held_without_prediction_symbols: list[str] | None  # 判定をしなかった日は None（#1530）
    stale_prediction_reason: str | None
    threshold_scale: float | None
    order_record_written: bool  # その回の記録を書きに行ったか。二重に書かない（#1548）


def _new_stats() -> OrderExecutionStats:
    """run_daily_orders の集計の初期値を返す。"""
    return {
        "buy_orders": 0,
        "sell_orders": 0,
        "short_orders": 0,
        "skipped": 0,
        "skipped_min_change": 0,
        "skip_reasons": {},
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
        "held_without_prediction_symbols": None,
        "stale_prediction_reason": None,
        "threshold_scale": None,
        "order_record_written": False,
    }


_MIN_CHANGE_KEYS = ("min_change", "sell_min_change")


def count_skip(stats: OrderExecutionStats, reason: str) -> None:
    """飛ばした候補を skipped と理由ごとの skip_reasons に1つ数える（#1765）。

    鍵は 16:30 の記録（RuleGateRecord）とそろえる。max_positions は保有上限、min_change は
    変動の小ささ（R-214）、sell_min_change は売りシグナルの変動の小ささ、no_price は現在値なし、
    low_confidence は確信度の見送り（R-308）、株数0は zero_qty_skip_key の鍵（#1748）。
    変動の小ささは従来どおり skipped_min_change にも数える。
    """
    stats["skipped"] += 1
    reasons = stats["skip_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1
    if reason in _MIN_CHANGE_KEYS:
        stats["skipped_min_change"] += 1
