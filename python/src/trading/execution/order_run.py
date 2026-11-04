"""run_daily_orders — 自動発注の入口。

手順の本体は runner.py の run_order_steps にある。手順が例外で途中で落ちた朝も、
記録（results/orders/<JST の日付>.json）を書いてから同じ例外を投げ直す（#1548）。
"""

from src.domain.ports import (
    MarketDataPort,
    NotificationPort,
    OrderRunSink,
    PredictionResultRepository,
    TradeDiffSink,
)
from src.trading.brokers.base import BrokerBase

from .gate_record import record_failed_run
from .runner import run_order_steps
from .stats import OrderExecutionStats, _new_stats


def run_daily_orders(
    broker: BrokerBase,
    *,
    order_run_sink: OrderRunSink,
    trade_diff_sink: TradeDiffSink,
    prediction_repo: PredictionResultRepository,
    market: str = "jp",
    mode: str = "paper",
    market_data: MarketDataPort | None = None,
    notifier: NotificationPort | None = None,
    prediction_saved_day: str | None = None,  # [3/6] の保存の終わりの JST の日付（#1394）
) -> OrderExecutionStats:
    """日次自動発注メインエントリーポイント。引数の意味は run_order_steps に書いてある。"""
    stats = _new_stats()
    try:
        return run_order_steps(
            stats,
            broker,
            order_run_sink=order_run_sink,
            trade_diff_sink=trade_diff_sink,
            prediction_repo=prediction_repo,
            market=market,
            mode=mode,
            market_data=market_data,
            notifier=notifier,
            prediction_saved_day=prediction_saved_day,
        )
    except Exception as exc:
        record_failed_run(stats, exc, market=market, mode=mode)
        raise
