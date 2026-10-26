"""ルールシグナルに基づくペーパートレード実行"""

from __future__ import annotations

from collections import Counter
from typing import Any

from config.settings import (
    CORRELATION_ENC_THRESHOLD,
    CORRELATION_WINDOW_DAYS,
    MAX_ORDERS_PER_RUN,
    MAX_POSITIONS,
    MAX_SECTOR_POSITIONS,
)
from src.domain.ports import TradeDiffSink
from src.domain.sector_cap import filter_by_sector_cap
from src.domain.trading_rules import get_lot_size
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.brokers.paper.paper_broker import PaperBroker
from src.trading.correlation_risk import evaluate_correlation_gate
from src.trading.execution.pending import _load_pending_exits, _log_pending_skips
from src.trading.risk_manager import RiskManager
from src.utils.data_path_utils import get_ticker
from src.utils.logger import get_logger
from src.utils.sector_constraints import get_symbol_sector

logger = get_logger(__name__)


def execute_rule_paper_trades(
    signals: list[dict[str, Any]],
    market: str,
    initial_budget_per_trade: float = 100_000,
    market_data_port=None,
    *,
    trade_diff_sink: TradeDiffSink,
    broker: BrokerBase | None = None,
) -> dict[str, int]:
    """
    シグナルに基づきペーパートレードを実行する。

    - BUY signal  → ポジション未保有の場合のみ買い注文
    - SELL signal → ポジション保有中の場合のみ売り注文

    Args:
        signals: run_rule_signal_pipeline の返り値
        market: マーケット識別子
        initial_budget_per_trade: 1銘柄あたりの最大投資額（概算）
        market_data_port: MarketDataPort 実装（呼び出し元から注入）
        trade_diff_sink: 約定乖離の記録先（TradeDiffSink 実装。合成ルートが必ず渡す）
        broker: 注文を出す先（省略時は PaperBroker。影の運用では ShadowBroker）

    Returns:
        {"buy_orders": int, "sell_orders": int, "skipped": int, "errors": int}
        errors は注文（株数の計算を含む）が例外で落ちた銘柄の数で、skipped に含めない。
    """
    if broker is None:
        broker = PaperBroker(
            market_data_port=market_data_port,
            trade_diff_sink=trade_diff_sink,
        )
    risk = RiskManager(broker, market=market)
    lot = get_lot_size(market)
    buy_orders = 0
    sell_orders = 0
    skipped = 0
    errors = 0

    # 保有を読めないまま空の保有で進めると、保有の数え直し・同じ銘柄の重ね買い・
    # 保有中の売りの取りこぼしが起きるため、注文を出さずにジョブを失敗させる。
    try:
        positions = broker.get_positions()
    except Exception:
        logger.error("[rule] 保有の読み込みに失敗したため発注を中止する", exc_info=True)
        raise
    held_symbols = {p["symbol"] for p in positions}

    # ML 経路（execution.run_daily_orders）と同等のリスク制御を適用する。
    # R-307: DD 基準値を発注前に更新し、当日発注可否ゲート（日次損失・連続損失・
    # 保有上限）を評価する。不合格時は新規買いを停止するが、決済（SELL）は許可する。
    risk.update_peak_balance()
    gate = risk.evaluate_trading_gate()
    if not gate.is_allowed:
        logger.warning(
            "[rule] リスクゲート不合格のため新規買いを停止: %s",
            gate.reason or "理由未設定",
        )

    # 8:50 の道（execution.runner）と同じく、相関ゲート不合格なら新規買いだけ止め、
    # 同一業種の上限は保有中の銘柄の業種も含めて数える（#1138）。
    buy_candidates = [
        item for item in signals if item["signal"] == 1 and item["symbol"] not in held_symbols
    ]
    sector_allowed: set[str] = set()
    if gate.is_allowed and buy_candidates:
        held_codes = sorted({s.replace(".T", "") for s in held_symbols})
        corr_gate = evaluate_correlation_gate(
            held_codes,
            market,
            window=CORRELATION_WINDOW_DAYS,
            enc_threshold=CORRELATION_ENC_THRESHOLD,
        )
        if corr_gate.is_allowed:
            selected = filter_by_sector_cap(
                buy_candidates,
                max_sector_positions=MAX_SECTOR_POSITIONS,
                sector_getter=lambda it: get_symbol_sector(market, str(it["symbol"])),
                initial_counts=Counter(get_symbol_sector(market, s) for s in held_codes),
            )
            sector_allowed = {str(it["symbol"]) for it in selected}
        else:
            logger.warning(
                "[rule] 相関ゲート不合格のため新規買いを停止: %s",
                corr_gate.reason or "ENC閾値未満",
            )

    position_count = len(held_symbols)  # 1ラン内で増える建玉数を追跡し MAX_POSITIONS を厳守
    buy_orders_this_run = 0
    # 朝の約定で据え置かれた SELL がある銘柄には2本目を出さない（#1169。8:50 の道と同じ確かめ）
    pending = _load_pending_exits(broker)
    pending_skipped: list[str] = []

    for item in signals:
        symbol = item["symbol"]
        signal = item["signal"]
        price = item.get("price") or 0.0

        try:
            if signal == 1 and symbol not in held_symbols:
                if not gate.is_allowed:
                    skipped += 1
                    continue
                if symbol not in sector_allowed:
                    logger.info("[rule] %s: 相関ゲートか同一業種の上限で新規買いスキップ", symbol)
                    skipped += 1
                    continue
                if position_count >= MAX_POSITIONS:
                    logger.info(
                        "[rule] %s: 保有上限 %d 到達のため新規買いスキップ", symbol, MAX_POSITIONS
                    )
                    skipped += 1
                    continue
                if buy_orders_this_run >= MAX_ORDERS_PER_RUN:
                    logger.info(
                        "[rule] %s: 1ラン発注上限 %d 到達のため新規買いスキップ",
                        symbol,
                        MAX_ORDERS_PER_RUN,
                    )
                    skipped += 1
                    continue

                # 残高連動のハーフ Kelly 基準で株数を算出（残高不足時は 0 株 → スキップ）。
                # initial_budget_per_trade は 1 銘柄あたりの投資上限として併用する（保守側）。
                qty = risk.calc_position_size(symbol, price)
                if price > 0 and initial_budget_per_trade > 0:
                    budget_cap = (int(initial_budget_per_trade / price) // lot) * lot
                    qty = min(qty, budget_cap)
                if qty <= 0:
                    logger.info("[rule] %s: 発注株数 0（残高不足 or サイズ上限）→ スキップ", symbol)
                    skipped += 1
                    continue

                ticker = get_ticker(market, symbol)
                broker.send_order(
                    symbol=ticker,
                    side=OrderSide.BUY,
                    qty=qty,
                    order_type=OrderType.MARKET,
                )
                buy_orders += 1
                buy_orders_this_run += 1
                position_count += 1
                logger.info(f"  BUY 注文: {symbol}  {qty}株  現在値={price:.0f}円")

            elif signal == -1 and symbol in held_symbols:
                if pending.has(symbol, OrderSide.SELL):
                    pending_skipped.append(symbol)
                    skipped += 1
                    continue
                pos = next((p for p in positions if p["symbol"] == symbol), None)
                qty = pos.get("qty", 1) if pos else 1
                ticker = get_ticker(market, symbol)
                broker.send_order(
                    symbol=ticker,
                    side=OrderSide.SELL,
                    qty=qty,
                    order_type=OrderType.MARKET,
                )
                sell_orders += 1
                logger.info(f"  SELL 注文: {symbol}  {qty}株  現在値={price:.0f}円")

            else:
                skipped += 1

        except Exception as exc:
            # 失敗はスキップと分けて数え、呼び出し元がジョブを失敗にできるようにする（#1075）。
            # ほかの銘柄の注文は止めない。
            logger.error(f"  {symbol} 注文失敗: {exc}", exc_info=True)
            errors += 1

    _log_pending_skips("ルールの売りシグナル", pending_skipped)
    logger.info(
        f"ペーパートレード実行: BUY={buy_orders}  SELL={sell_orders}  "
        f"スキップ={skipped}  失敗={errors}"
    )
    return {
        "buy_orders": buy_orders,
        "sell_orders": sell_orders,
        "skipped": skipped,
        "errors": errors,
    }
