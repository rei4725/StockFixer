"""ホライズン決済: target_exit_date を過ぎたペーパーの保有を成行で決済する。

daily.run_horizon_exit_check から呼ばれる（モードの判定は呼び出し元で行う）。
"""

from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# paper_orders の行は決済した後も残るため、銘柄ごとに今の保有を作った
# 最後の filled の約定の期限だけを見る (Issue #945)。買い持ちと空売りは side ごとに分ける (#1145)
_EXPIRED_SQL = """
    SELECT symbol
    FROM (
        SELECT DISTINCT ON (symbol) symbol, target_exit_date
        FROM paper_orders
        WHERE side = %s
          AND status = 'filled'
        ORDER BY symbol, filled_at DESC NULLS LAST, created_at DESC, order_id DESC
    ) AS latest_entry
    WHERE target_exit_date IS NOT NULL
      AND CAST(target_exit_date AS VARCHAR) < %s
    ORDER BY symbol
"""


def _find_position(positions: list[dict[str, Any]], symbol: str) -> dict[str, Any] | None:
    return next((p for p in positions if p["symbol"].replace(".T", "") == symbol), None)


def exit_expired_positions() -> list[str]:
    """期限を過ぎた買い持ちは成行売り、空売りは SHORT_COVER の成行で返済する。

    SL/TP によって既に決済済みのポジションは保有に存在しないため
    自然に除外される（SL/TP 優先）。
    買い持ちは銘柄ごとに最後の filled BUY、空売りは最後の filled SHORT の
    target_exit_date だけを見る。約定前の SELL がある銘柄の買い持ちと、
    約定前の SHORT_COVER がある銘柄の空売りは飛ばす。

    Returns:
        決済の注文に失敗した銘柄コードのリスト。
    """
    from src.infrastructure.discord_notification_adapter import DiscordNotificationAdapter
    from src.infrastructure.persistence.trade_diff_repository import PostgresTradeDiffSink
    from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
    from src.orchestration.jobs.common import today_jst_iso
    from src.trading.brokers.account.factory import make_order_broker
    from src.trading.brokers.base import OrderSide
    from src.trading.execution.pending import _log_pending_skips, _read_pending_exits
    from src.trading.position_valuation import coverable_short_qty
    from src.utils.db._connection import _db_connection

    today_str = today_jst_iso()  # target_exit_date は保有の最後の JST の日 (#949)
    with _db_connection() as con:
        long_rows = con.execute(_EXPIRED_SQL, [int(OrderSide.BUY), today_str]).fetchall()
        short_rows = con.execute(_EXPIRED_SQL, [int(OrderSide.SHORT), today_str]).fetchall()

    long_symbols = [row[0] for row in long_rows]
    short_symbols = [row[0] for row in short_rows]
    if not long_symbols and not short_symbols:
        logger.info("[horizon_exit] 期限切れポジションなし")
        return []

    logger.info(
        "[horizon_exit] 期限切れポジション対象: 買い持ち=%s 空売り=%s", long_symbols, short_symbols
    )
    broker = make_order_broker(
        job="horizon",
        market_data_port=YFinanceMarketDataAdapter(),
        trade_diff_sink=PostgresTradeDiffSink(),
        notifier=DiscordNotificationAdapter(),
    )
    # 送った注文は約定まで保有を変えないので、保有の一覧はループの前に1回だけ読む（#1095）
    positions = broker.get_positions() if long_symbols else []
    short_positions: list[dict[str, Any]] = []
    short_read_failed = False
    if short_symbols:
        # 読めない日は空売りを0件として黙って飛ばさず、買い持ちを決済した後に失敗にする (#1451)
        try:
            short_positions = broker.get_short_positions()
        except Exception:
            logger.error("[horizon_exit] 空売りの保有を読めない: %s", short_symbols, exc_info=True)
            short_read_failed = True

    # 保有の一覧は約定前の SELL・SHORT_COVER を引かないため、重ねると返済の二重発注になる (#1168)
    pending = _read_pending_exits()
    exits: list[tuple[str, OrderSide, dict[str, Any] | None]] = []
    skipped: list[str] = []
    failed: list[str] = []
    for symbol in long_symbols:
        if pending.has(symbol, OrderSide.SELL):
            skipped.append(symbol)
            continue
        exits.append((symbol, OrderSide.SELL, _find_position(positions, symbol)))
    for symbol in short_symbols:
        if pending.has(symbol, OrderSide.SHORT_COVER):
            skipped.append(symbol)
            continue
        if short_read_failed:
            failed.append(symbol)
            continue
        exits.append((symbol, OrderSide.SHORT_COVER, _find_position(short_positions, symbol)))
    _log_pending_skips("horizon_exit", skipped)

    exited: list[str] = []
    for symbol, side, pos in exits:
        if pos is None:
            continue
        # 空売りは約定待ちの SHORT の分を返さない (#1252)
        qty = coverable_short_qty(pos) if side == OrderSide.SHORT_COVER else pos.get("qty", 0)
        if qty <= 0:
            continue
        try:
            broker.send_order(symbol, side, qty)
            logger.info("[horizon_exit] ホライズン決済: %s %s %d株", symbol, side.name, qty)
            exited.append(symbol)
        except Exception as e:
            logger.error("[horizon_exit] 決済失敗: %s: %s", symbol, e, exc_info=True)
            failed.append(symbol)

    logger.info("[horizon_exit] ホライズン決済完了: %d 件", len(exited))
    return failed
