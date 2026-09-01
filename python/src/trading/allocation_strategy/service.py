"""配分戦略(TQQQ 80% / 短期債 20%目安、約2年ごとリバランス)のペーパートレード実行ロジック。"""

from datetime import datetime, timedelta
from typing import Optional

from src.domain.ports import MarketDataPort
from src.trading.allocation_strategy.repository import get_latest_snapshot, insert_snapshot
from src.trading.allocation_strategy.types import RebalanceOutcome
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_allocation_rebalance(market_data_port: MarketDataPort) -> Optional[RebalanceOutcome]:
    """配分戦略の初期建玉作成、または期日到来時のリバランスを実行する。

    Args:
        market_data_port: 価格取得に使うポート実装。BC(trading)はinfrastructure/
            他BCを直接importできないレイヤー規約のため、具象アダプタ
            (YFinanceMarketDataAdapter等)の生成と注入は呼び出し側(orchestration層)
            の責務とする。

    まだ状態が無ければ初期建玉を作成する(action="initial")。
    既に状態があり、前回実行から ALLOCATION_STRATEGY_REBALANCE_YEARS 年以上
    (365日換算の近似)経過していればリバランスする(action="rebalance")。
    それ以外(未到来、または本日既に実行済み)は None を返し何もしない。
    価格取得に失敗した場合(0.0以下)も None を返し、状態は書き込まない。
    """
    from config.settings import (
        ALLOCATION_STRATEGY_BOND_SYMBOL,
        ALLOCATION_STRATEGY_INITIAL_CAPITAL,
        ALLOCATION_STRATEGY_REBALANCE_YEARS,
        ALLOCATION_STRATEGY_TQQQ_RATIO,
        ALLOCATION_STRATEGY_TQQQ_SYMBOL,
    )

    latest = get_latest_snapshot()
    now = datetime.now()

    if latest is not None:
        if latest.executed_at.date() == now.date():
            logger.info("配分戦略: 本日は既に実行済みのためスキップ")
            return None
        due_at = latest.executed_at + timedelta(days=365 * ALLOCATION_STRATEGY_REBALANCE_YEARS)
        if now < due_at:
            logger.info(
                "配分戦略: リバランス期日未到来のためスキップ（次回予定: %s）",
                due_at.date(),
            )
            return None

    tqqq_price = market_data_port.get_latest_price(ALLOCATION_STRATEGY_TQQQ_SYMBOL)
    shy_price = market_data_port.get_latest_price(ALLOCATION_STRATEGY_BOND_SYMBOL)
    if tqqq_price <= 0 or shy_price <= 0:
        logger.error(
            "配分戦略: 価格取得失敗のため中止（tqqq_price=%s, shy_price=%s）",
            tqqq_price,
            shy_price,
        )
        return None

    if latest is None:
        action = "initial"
        tqqq_qty_before = 0.0
        shy_qty_before = 0.0
        cash_before = ALLOCATION_STRATEGY_INITIAL_CAPITAL
        total_value = ALLOCATION_STRATEGY_INITIAL_CAPITAL
    else:
        action = "rebalance"
        tqqq_qty_before = latest.tqqq_qty_after
        shy_qty_before = latest.shy_qty_after
        cash_before = latest.cash_after
        total_value = tqqq_qty_before * tqqq_price + shy_qty_before * shy_price + cash_before

    tqqq_qty_after = (total_value * ALLOCATION_STRATEGY_TQQQ_RATIO) / tqqq_price
    shy_target_value = total_value * (1 - ALLOCATION_STRATEGY_TQQQ_RATIO)
    shy_qty_after = shy_target_value / shy_price
    cash_after = total_value - (tqqq_qty_after * tqqq_price + shy_qty_after * shy_price)

    insert_snapshot(
        action=action,
        tqqq_price=tqqq_price,
        shy_price=shy_price,
        tqqq_qty_before=tqqq_qty_before,
        shy_qty_before=shy_qty_before,
        cash_before=cash_before,
        tqqq_qty_after=tqqq_qty_after,
        shy_qty_after=shy_qty_after,
        cash_after=cash_after,
    )

    return RebalanceOutcome(
        action=action,
        tqqq_price=tqqq_price,
        shy_price=shy_price,
        tqqq_qty_before=tqqq_qty_before,
        shy_qty_before=shy_qty_before,
        cash_before=cash_before,
        tqqq_qty_after=tqqq_qty_after,
        shy_qty_after=shy_qty_after,
        cash_after=cash_after,
    )
