"""配分戦略(TQQQ/短期債)ペーパートレードの日次エクイティ系列を再構成する。

allocation_rebalance_log の各スナップショットは保有数量が変化しない区間の
開始点を表す。区間内は保有数量固定として日次終値でマークトゥマーケットする。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.domain.ports import MarketDataPort
from src.trading.allocation_strategy.repository import list_snapshots
from src.trading.allocation_strategy.types import AllocationSnapshot
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_allocation_equity_series(
    snapshots: list[AllocationSnapshot],
    tqqq_prices: pd.Series,
    shy_prices: pd.Series,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """スナップショット履歴と日次終値から日次エクイティ系列を構成する（純粋ロジック）。

    Args:
        snapshots: 実行日時昇順のスナップショット一覧
        tqqq_prices: TQQQ の日次終値 Series
        shy_prices: SHY の日次終値 Series
        index: 評価日付の DatetimeIndex（昇順・正規化済み）

    Returns:
        index に沿った評価額 Series（snapshots か index が空なら空 Series）
    """
    if not snapshots or index.empty:
        return pd.Series(dtype=float)

    tqqq_aligned = tqqq_prices.reindex(index).ffill().bfill()
    shy_aligned = shy_prices.reindex(index).ffill().bfill()

    boundaries = [pd.Timestamp(s.executed_at).normalize() for s in snapshots]

    equity = pd.Series(index=index, dtype=float)
    for day in index:
        # day 時点で有効な最新のスナップショット（未来分は対象外）
        active = snapshots[0]
        for snap, boundary in zip(snapshots, boundaries):
            if boundary <= day:
                active = snap
            else:
                break
        equity[day] = (
            active.tqqq_qty_after * tqqq_aligned[day]
            + active.shy_qty_after * shy_aligned[day]
            + active.cash_after
        )

    return equity


def _load_price_series(
    market_data_port: MarketDataPort, symbol: str, start: pd.Timestamp, end: pd.Timestamp
) -> Optional[pd.Series]:
    """symbol の日次終値 Series を取得する（取得不能時 None）。"""
    try:
        df = market_data_port.get_stock_data(
            symbol, "us", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        )
    except Exception as e:
        logger.warning("配分戦略: %s の価格取得失敗: %s", symbol, e)
        return None
    if df is None or df.empty or "Close" not in df.columns:
        return None
    series = df["Close"].copy()
    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
    return series


def get_allocation_equity_curve(market_data_port: MarketDataPort, days: int = 180) -> pd.Series:
    """配分戦略ペーパートレードの日次エクイティ系列を返す。

    Args:
        market_data_port: 価格取得に使うポート実装。BC(trading)はinfrastructure/
            他BCを直接importできないレイヤー規約のため、具象アダプタの生成と注入は
            呼び出し側(orchestration層)の責務とする。
        days: 評価対象の直近日数

    Returns:
        日次評価額 Series（スナップショットが1件も無い場合は空 Series）
    """
    from config.settings import ALLOCATION_STRATEGY_BOND_SYMBOL, ALLOCATION_STRATEGY_TQQQ_SYMBOL

    snapshots = list_snapshots()
    if not snapshots:
        logger.info("配分戦略のスナップショットがないためエクイティ系列は空")
        return pd.Series(dtype=float)

    end = pd.Timestamp(datetime.now().date())
    first = pd.Timestamp(snapshots[0].executed_at).normalize()
    start = max(first, end - timedelta(days=days))
    index = pd.bdate_range(start=start, end=end)

    tqqq_prices = _load_price_series(market_data_port, ALLOCATION_STRATEGY_TQQQ_SYMBOL, start, end)
    shy_prices = _load_price_series(market_data_port, ALLOCATION_STRATEGY_BOND_SYMBOL, start, end)
    if tqqq_prices is None or shy_prices is None:
        logger.warning("配分戦略: 価格データ取得不能のためエクイティ系列は空")
        return pd.Series(dtype=float)

    return build_allocation_equity_series(snapshots, tqqq_prices, shy_prices, index)
