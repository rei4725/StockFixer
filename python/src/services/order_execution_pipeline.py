"""
OrderExecutionPipeline — 自動発注オーケストレーター

DuckDB の最新予測結果を読み込み、RiskManager のゲートチェックを通過した
銘柄に対して Broker 経由で成行注文を発注する。

取引フロー:
    1. prediction_results から当日予測 Top N シグナルを取得
    2. RiskManager.is_trading_allowed() で当日取引可否を確認
    3. 保有済みポジションを確認し重複買いを回避
    4. RiskManager.calc_position_size() で発注株数を算出
    5. broker.send_order() で成行注文
    6. orders テーブルに記録
"""

import uuid
from typing import TypedDict

import pandas as pd

from config.settings import BUY_THRESHOLD, MAX_ORDERS_PER_RUN, MAX_SECTOR_POSITIONS
from src.brokers.base import BrokerBase, OrderSide, OrderType
from src.domain.types import TradingGateStatus
from src.services.prediction_pipeline import get_optimal_params
from src.services.risk_manager import RiskManager
from src.utils.db import upsert_paper_real_diff
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger
from src.utils.sector_constraints import filter_by_sector_cap, get_symbol_sector

logger = get_logger(__name__)

# 発注対象スコアの閾値
# BUY_THRESHOLD / SELL_THRESHOLD / MAX_ORDERS_PER_RUN は config/settings.py から取得
_THRESHOLD_SCALE_MIN = 0.5
_THRESHOLD_SCALE_MAX = 2.0
_THRESHOLD_SCALE_MIN_ROWS = 5


class OrderExecutionStats(TypedDict):
    buy_orders: int
    sell_orders: int
    skipped: int
    errors: int
    trading_stopped: bool
    stop_reason: str | None
    reason_code: str | None
    daily_loss: float
    daily_loss_limit: float | None


def _load_latest_predictions(market: str) -> pd.DataFrame:
    """
    DuckDB から当日の最新予測結果を取得する。

    Returns:
        columns: market, symbol, predicted_at, current_price, diff_ratio,
        confidence_ratio (desc order)
    """
    with _db_connection() as con:
        return con.execute(
            """
            WITH latest AS (
                SELECT market, symbol, MAX(predicted_at) AS latest_at
                FROM prediction_results
                WHERE market = ?
                GROUP BY market, symbol
            )
            SELECT
                pr.market,
                pr.symbol,
                pr.predicted_at,
                pr.current_price,
                pr.diff_ratio,
                pr.confidence_ratio
            FROM prediction_results pr
            JOIN latest l
              ON pr.market = l.market AND pr.symbol = l.symbol AND pr.predicted_at = l.latest_at
            WHERE pr.diff_ratio IS NOT NULL
            ORDER BY pr.diff_ratio DESC
            """,
            [market],
        ).df()


def _compute_market_threshold_scale(predictions: pd.DataFrame) -> float:
    """当日の予測分布から実運用の閾値スケールを決定する。"""
    if predictions.empty or "diff_ratio" not in predictions.columns:
        return 1.0

    abs_diff = pd.to_numeric(predictions["diff_ratio"], errors="coerce").abs().dropna()
    if len(abs_diff) < _THRESHOLD_SCALE_MIN_ROWS:
        return 1.0

    median_abs = float(abs_diff.median())
    dispersion = float(abs_diff.std())
    if median_abs <= 0 or pd.isna(dispersion) or dispersion <= 0:
        return 1.0

    raw_scale = dispersion / median_abs
    return float(min(_THRESHOLD_SCALE_MAX, max(_THRESHOLD_SCALE_MIN, raw_scale)))


def _resolve_base_threshold(market: str, symbol: str) -> float:
    """銘柄別の最適閾値を取得し、未設定時は既定値にフォールバックする。"""
    params = get_optimal_params(market, symbol)
    raw_threshold = params.get("threshold") if params else None
    try:
        threshold = abs(float(raw_threshold)) if raw_threshold is not None else 0.0
    except (TypeError, ValueError):
        threshold = 0.0

    if threshold <= 0:
        threshold = abs(BUY_THRESHOLD)
    return threshold


def _attach_dynamic_thresholds(predictions: pd.DataFrame) -> pd.DataFrame:
    """予測結果に実運用用の動的閾値列を付与する。"""
    if predictions.empty:
        return predictions.copy()

    scale = _compute_market_threshold_scale(predictions)
    threshold_cache: dict[tuple[str, str], float] = {}

    def _get_threshold(row: pd.Series) -> float:
        key = (str(row["market"]), str(row["symbol"]))
        if key not in threshold_cache:
            threshold_cache[key] = _resolve_base_threshold(*key)
        return threshold_cache[key]

    enriched = predictions.copy()
    enriched["base_threshold"] = enriched.apply(_get_threshold, axis=1)
    enriched["threshold_scale"] = scale
    enriched["effective_buy_threshold"] = enriched["base_threshold"] * scale
    enriched["effective_sell_threshold"] = -enriched["effective_buy_threshold"]
    return enriched


def _apply_buy_sector_limit(
    predictions: pd.DataFrame,
    max_sector_positions: int = MAX_SECTOR_POSITIONS,
) -> pd.DataFrame:
    """買い候補に同一セクター上限を適用する。"""
    if predictions.empty or max_sector_positions <= 0:
        return predictions.copy()

    ordered_rows = list(predictions.sort_values("diff_ratio", ascending=False).iterrows())
    sector_cache: dict[int, str] = {}

    def _sector_getter(item: tuple[int, pd.Series]) -> str:
        idx, row = item
        if idx not in sector_cache:
            sector_cache[idx] = get_symbol_sector(str(row["market"]), str(row["symbol"]))
        return sector_cache[idx]

    selected_rows = filter_by_sector_cap(
        ordered_rows,
        max_sector_positions=max_sector_positions,
        sector_getter=_sector_getter,
    )
    if not selected_rows:
        return predictions.iloc[0:0].copy()

    selected_index = [idx for idx, _ in selected_rows]
    limited = predictions.loc[selected_index].copy()
    limited["sector"] = [sector_cache[idx] for idx, _ in selected_rows]
    return limited


def _get_held_symbols(broker: BrokerBase) -> set[str]:
    """保有中の銘柄コードセットを返す"""
    positions = broker.get_positions()
    return {p["symbol"].replace(".T", "") for p in positions if p.get("qty", 0) > 0}


def _link_paper_order_metadata(
    order_id: str,
    market: str,
    predicted_at: str,
    signal_price: float,
) -> None:
    with _db_connection() as con:
        con.execute(
            """
            UPDATE paper_orders
            SET market = ?, predicted_at = ?, signal_price = ?
            WHERE order_id = ?
            """,
            [market, predicted_at, signal_price, order_id],
        )


def _sync_live_execution_diffs(broker: BrokerBase) -> None:
    for order in broker.get_orders():
        order_id = str(order.get("order_id") or "")
        price = order.get("price")
        if not order_id or price in (None, ""):
            continue
        try:
            actual_price = float(price)
        except (TypeError, ValueError):
            continue
        if actual_price <= 0:
            continue

        with _db_connection() as con:
            row = con.execute(
                """
                SELECT market, symbol, predicted_at, side, signal_price
                FROM paper_real_diff
                WHERE real_order_id = ?
                """,
                [order_id],
            ).fetchone()
        if row is None:
            continue

        upsert_paper_real_diff(
            market=str(row[0]),
            symbol=str(row[1]),
            predicted_at=str(row[2]),
            side=int(row[3]),
            signal_price=float(row[4] or 0.0),
            mode="live",
            order_id=order_id,
            actual_price=actual_price,
        )


def _record_order(
    market: str,
    predicted_at: str,
    symbol: str,
    side: OrderSide,
    qty: int,
    signal_price: float,
    order_result: dict,
    broker: BrokerBase,
    mode: str,
) -> None:
    """注文結果を orders テーブルに保存する"""
    with _db_connection() as con:
        con.execute(
            """
            INSERT INTO orders
                (order_id, symbol, side, qty, price, order_type, status, broker, mode, created_at)
            VALUES (?, ?, ?, ?, 0.0, 10, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                order_result.get("order_id", str(uuid.uuid4())[:12]),
                symbol,
                int(side),
                qty,
                order_result.get("status", "unknown"),
                broker.broker_name,
                mode,
            ],
        )

    order_id = str(order_result.get("order_id", ""))
    if mode == "paper" and order_id:
        _link_paper_order_metadata(order_id, market, predicted_at, signal_price)

    fill_price_raw = order_result.get("fill_price")
    fill_price = float(fill_price_raw) if isinstance(fill_price_raw, (int, float)) else None

    upsert_paper_real_diff(
        market=market,
        symbol=symbol,
        predicted_at=predicted_at,
        side=int(side),
        signal_price=signal_price,
        mode=mode,
        order_id=order_id,
        actual_price=fill_price,
    )


def run_daily_orders(
    broker: BrokerBase,
    market: str = "jp",
    mode: str = "paper",
) -> OrderExecutionStats:
    """
    日次自動発注メインエントリーポイント。

    Args:
        broker: BrokerBase 実装（KabuBroker or PaperBroker）
        market: 対象マーケット（"jp" = 東証）
        mode: "paper" or "live"

    Returns:
        {"buy_orders": int, "sell_orders": int, "skipped": int, "errors": int}
    """
    logger.info(f"=== 自動発注開始: market={market} mode={mode} broker={broker.broker_name} ===")

    risk = RiskManager(broker)
    stats: OrderExecutionStats = {
        "buy_orders": 0,
        "sell_orders": 0,
        "skipped": 0,
        "errors": 0,
        "trading_stopped": False,
        "stop_reason": None,
        "reason_code": None,
        "daily_loss": 0.0,
        "daily_loss_limit": None,
    }

    # --- 当日取引可否チェック ---
    gate_status: TradingGateStatus = risk.evaluate_trading_gate()
    if not gate_status.is_allowed:
        logger.warning(
            "[exec] リスクチェック不合格 → 本日の発注をスキップ: %s",
            gate_status.reason or "理由未設定",
        )
        stats.update(
            {
                "trading_stopped": gate_status.stop_active,
                "stop_reason": gate_status.reason,
                "reason_code": gate_status.reason_code,
                "daily_loss": gate_status.daily_loss,
                "daily_loss_limit": gate_status.daily_loss_limit,
            }
        )
        return stats

    predictions = _load_latest_predictions(market)
    if predictions.empty:
        logger.warning("[exec] 予測結果が存在しません。先に run_predict.py を実行してください。")
        stats.update(
            {
                "daily_loss": gate_status.daily_loss,
                "daily_loss_limit": gate_status.daily_loss_limit,
            }
        )
        return stats
    if "predicted_at" not in predictions.columns:
        predictions = predictions.copy()
        predictions["predicted_at"] = ""

    predictions = _attach_dynamic_thresholds(predictions)
    threshold_scale = float(predictions["threshold_scale"].iloc[0])
    logger.info(
        "[exec] 動的閾値適用: scale=%.3f buy_range=%.3f%%-%.3f%%",
        threshold_scale,
        predictions["effective_buy_threshold"].min() * 100,
        predictions["effective_buy_threshold"].max() * 100,
    )

    held_symbols = _get_held_symbols(broker)
    stats.update(
        {
            "daily_loss": gate_status.daily_loss,
            "daily_loss_limit": gate_status.daily_loss_limit,
        }
    )

    # --- 決済シグナル: 保有株で売りシグナルが出ているものを先にクローズ ---
    sell_signals = predictions[
        (predictions["symbol"].isin(held_symbols))
        & (predictions["diff_ratio"] <= predictions["effective_sell_threshold"])
    ]
    for _, row in sell_signals.iterrows():
        symbol = row["symbol"]
        try:
            # 保有全数を成行売り
            pos = next(
                (p for p in broker.get_positions() if p["symbol"].replace(".T", "") == symbol), None
            )
            if pos is None or pos["qty"] <= 0:
                continue
            qty = pos["qty"]
            result = broker.send_order(symbol, OrderSide.SELL, qty, order_type=OrderType.MARKET)
            _record_order(
                market=str(row["market"]),
                predicted_at=str(row["predicted_at"]),
                symbol=symbol,
                side=OrderSide.SELL,
                qty=qty,
                signal_price=float(row.get("current_price") or 0.0),
                order_result=result,
                broker=broker,
                mode=mode,
            )
            logger.info(
                f"[exec] 売り発注: {symbol} {qty}株 "
                f"(予測変化率={row['diff_ratio']:.3%}, 閾値={row['effective_sell_threshold']:.3%})"
            )
            stats["sell_orders"] += 1
        except Exception as e:
            logger.error(f"[exec] 売り注文エラー ({symbol}): {e}", exc_info=True)
            stats["errors"] += 1

    # --- 新規買いシグナル ---
    buy_candidates = predictions[
        (~predictions["symbol"].isin(held_symbols))
        & (predictions["diff_ratio"] >= predictions["effective_buy_threshold"])
    ]
    buy_signals = _apply_buy_sector_limit(buy_candidates).head(MAX_ORDERS_PER_RUN)

    for _, row in buy_signals.iterrows():
        symbol = row["symbol"]
        current_price = float(row.get("current_price") or 0)

        if current_price <= 0:
            logger.warning(f"[exec] {symbol}: 現在値が取得できないためスキップ")
            stats["skipped"] += 1
            continue

        qty = risk.calc_position_size(
            symbol,
            current_price,
            confidence_ratio=float(row.get("confidence_ratio") or 1.0),
        )
        if qty <= 0:
            logger.info(f"[exec] {symbol}: 発注株数 0 → スキップ（残高不足or上限）")
            stats["skipped"] += 1
            continue

        try:
            result = broker.send_order(symbol, OrderSide.BUY, qty, order_type=OrderType.MARKET)
            _record_order(
                market=str(row["market"]),
                predicted_at=str(row["predicted_at"]),
                symbol=symbol,
                side=OrderSide.BUY,
                qty=qty,
                signal_price=current_price,
                order_result=result,
                broker=broker,
                mode=mode,
            )
            logger.info(
                f"[exec] 買い発注: {symbol} {qty}株 @ 成行 "
                f"(予測変化率={row['diff_ratio']:.3%}, 閾値={row['effective_buy_threshold']:.3%}, "
                f"sector={row.get('sector', 'N/A')})"
            )
            stats["buy_orders"] += 1
        except Exception as e:
            logger.error(f"[exec] 買い注文エラー ({symbol}): {e}", exc_info=True)
            stats["errors"] += 1

    logger.info(
        f"=== 自動発注完了: 買い={stats['buy_orders']} 売り={stats['sell_orders']} "
        f"スキップ={stats['skipped']} エラー={stats['errors']} ==="
    )
    if mode == "live":
        _sync_live_execution_diffs(broker)
    return stats
