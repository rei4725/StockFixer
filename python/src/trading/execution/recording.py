"""注文結果の記録と live 約定差分の同期。"""

from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from src.domain.jp_market_holidays import is_jp_business_day
from src.domain.ports import TradeDiffSink
from src.domain.types import TradeDiffRecord
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.utils.db import db_connection
from src.utils.japan_time import now_jst
from src.utils.logger import get_logger

from .stats import OrderExecutionStats

logger = get_logger(__name__)


def _nth_business_day(order_day: date, horizon: int) -> date:
    """発注日を1日目として horizon 番目の東証の営業日を返す（horizon 1 は発注日）。"""
    day = order_day
    for _ in range(horizon - 1):
        day += timedelta(days=1)
        while not is_jp_business_day(day):
            day += timedelta(days=1)
    return day


def _link_paper_order_metadata(
    order_id: str,
    market: str,
    predicted_at: str,
    signal_price: float,
    horizon: int | None = None,
    target_exit_date: str | None = None,
) -> None:
    with db_connection() as con:
        con.execute(
            """
            UPDATE paper_orders
            SET market = %s, predicted_at = %s, signal_price = %s,
                horizon = %s, target_exit_date = %s
            WHERE order_id = %s
            """,
            [market, predicted_at, signal_price, horizon, target_exit_date, order_id],
        )


def _sync_live_execution_diffs(broker: BrokerBase, trade_diff_sink: TradeDiffSink) -> None:
    for order in broker.get_orders():
        order_id = str(order.get("order_id") or "")
        price = order.get("price")
        if not order_id or price in (None, ""):
            continue
        try:
            actual_price = float(price)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if actual_price <= 0:
            continue

        with db_connection() as con:
            row = con.execute(
                """
                SELECT market, symbol, predicted_at, side, signal_price
                FROM paper_real_diff
                WHERE real_order_id = %s
                """,
                [order_id],
            ).fetchone()
        if row is None:
            continue

        trade_diff_sink.record(
            TradeDiffRecord(
                market=str(row[0]),
                symbol=str(row[1]),
                predicted_at=str(row[2]),
                side=int(row[3]),
                signal_price=float(row[4] or 0.0),
                mode="live",
                order_id=order_id,
                actual_price=actual_price,
            )
        )


def _record_order(
    market: str,
    predicted_at: str,
    symbol: str,
    side: OrderSide,
    qty: int,
    signal_price: float,
    order_price: float,
    order_type: OrderType,
    order_result: dict[str, Any],
    broker: BrokerBase,
    mode: str,
    *,
    trade_diff_sink: TradeDiffSink,
    order_session: str = "open",
    split_ratio: float = 1.0,
    horizon: int | None = None,
) -> None:
    """注文結果に付随するメタデータ（paper_orders 補完・paper/real diff）を記録する。

    実際の注文行そのものは broker.send_order() が記録する
    （PaperBroker は paper_orders、KabuBroker は Kabu Station 側）。
    ここでは PaperBroker が書いた paper_orders 行に predicted_at/signal_price/
    horizon/target_exit_date を補完し（mode=="paper" のみ）、paper/real 価格差分
    （paper_real_diff）を記録する（#586: かつて存在しない `orders` テーブルへの
    INSERT がここにあり、例外で以下の処理全体が握りつぶされていた）。
    """
    order_id = str(order_result.get("order_id", ""))
    if mode == "paper" and order_id:
        target_exit_date: str | None = None
        if horizon is not None and side in (OrderSide.BUY, OrderSide.SHORT):
            # 保有の最後の JST の日（発注日を1日目とする horizon 番目の東証の営業日）。
            # 決済はその次の日の朝 (#949, #1119)
            target_exit_date = _nth_business_day(now_jst().date(), horizon).isoformat()
        try:
            _link_paper_order_metadata(
                order_id, market, predicted_at, signal_price, horizon, target_exit_date
            )
        except Exception:
            if side in (OrderSide.BUY, OrderSide.SHORT):
                # 期限の無い保有を残さないよう、期限を書けない新規の注文は取り消す (#1104)
                logger.error("[exec] 注文の期限を書けず取り消す: order_id=%s", order_id)
                broker.cancel_order(order_id)
            raise

    fill_price_raw = order_result.get("fill_price")
    fill_price = float(fill_price_raw) if isinstance(fill_price_raw, (int, float)) else None

    trade_diff_sink.record(
        TradeDiffRecord(
            market=market,
            symbol=symbol,
            predicted_at=predicted_at,
            side=int(side),
            signal_price=signal_price,
            mode=mode,
            order_id=order_id,
            actual_price=fill_price,
            order_session=order_session,
            split_ratio=split_ratio,
        )
    )


def _recorder(record: Callable[..., None], stats: OrderExecutionStats) -> Callable[..., None]:
    """send_order が通った後の付帯の記録を、失敗しても数えを止めない形で包む（#1104）。

    注文はもう出ているため、記録の失敗で呼び出し元の数え（保有の上限・SL/TP の除外）を
    止めない。失敗は order_id を添えて ERROR に出し、errors に数える。
    record は呼び出し元のモジュールの _record_order（テストが差し替える名前）を渡す。
    """

    def _record_after_send(**kwargs: Any) -> None:
        try:
            record(**kwargs)
        except Exception:
            logger.error(
                "[exec] 注文は出たが付帯の記録に失敗 (%s order_id=%s)",
                kwargs["symbol"],
                kwargs["order_result"].get("order_id", ""),
                exc_info=True,
            )
            stats["errors"] += 1

    return _record_after_send
