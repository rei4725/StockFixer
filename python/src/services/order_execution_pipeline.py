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

from src.brokers.base import BrokerBase, OrderSide, OrderType
from src.domain.types import TradingGateStatus
from src.services.risk_manager import RiskManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 発注対象スコアの閾値（diff_ratio が この値以上でBuy対象）
BUY_THRESHOLD = 0.005  # +0.5%
SELL_THRESHOLD = -0.005  # -0.5%
# 1回の実行で発注する最大銘柄数
MAX_ORDERS_PER_RUN = 5


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


def _get_con():
    from src.utils.db import get_connection

    return get_connection()


def _load_latest_predictions(market: str) -> pd.DataFrame:
    """
    DuckDB から当日の最新予測結果を取得する。

    Returns:
        columns: market, symbol, current_price, diff_ratio (desc order)
    """
    con = _get_con()
    df = con.execute(
        """
        WITH latest AS (
            SELECT market, symbol, MAX(predicted_at) AS latest_at
            FROM prediction_results
            WHERE market = ?
            GROUP BY market, symbol
        )
        SELECT pr.market, pr.symbol, pr.current_price, pr.diff_ratio
        FROM prediction_results pr
        JOIN latest l
          ON pr.market = l.market AND pr.symbol = l.symbol AND pr.predicted_at = l.latest_at
        WHERE pr.diff_ratio IS NOT NULL
        ORDER BY pr.diff_ratio DESC
        """,
        [market],
    ).df()
    return df


def _get_held_symbols(broker: BrokerBase) -> set[str]:
    """保有中の銘柄コードセットを返す"""
    positions = broker.get_positions()
    return {p["symbol"].replace(".T", "") for p in positions if p.get("qty", 0) > 0}


def _record_order(
    symbol: str,
    side: OrderSide,
    qty: int,
    order_result: dict,
    broker: BrokerBase,
    mode: str,
) -> None:
    """注文結果を orders テーブルに保存する"""
    con = _get_con()
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

    held_symbols = _get_held_symbols(broker)
    stats.update(
        {
            "daily_loss": gate_status.daily_loss,
            "daily_loss_limit": gate_status.daily_loss_limit,
        }
    )

    # --- 決済シグナル: 保有株で売りシグナルが出ているものを先にクローズ ---
    sell_signals = predictions[
        (predictions["symbol"].isin(held_symbols)) & (predictions["diff_ratio"] <= SELL_THRESHOLD)
    ]
    for _, row in sell_signals.iterrows():
        symbol = row["symbol"]
        current_price = float(row.get("current_price") or 0)
        try:
            # 保有全数を成行売り
            pos = next(
                (p for p in broker.get_positions() if p["symbol"].replace(".T", "") == symbol), None
            )
            if pos is None or pos["qty"] <= 0:
                continue
            qty = pos["qty"]
            result = broker.send_order(symbol, OrderSide.SELL, qty, order_type=OrderType.MARKET)
            _record_order(symbol, OrderSide.SELL, qty, result, broker, mode)
            logger.info(f"[exec] 売り発注: {symbol} {qty}株 (予測変化率={row['diff_ratio']:.3%})")
            stats["sell_orders"] += 1
        except Exception as e:
            logger.error(f"[exec] 売り注文エラー ({symbol}): {e}", exc_info=True)
            stats["errors"] += 1

    # --- 新規買いシグナル ---
    buy_signals = predictions[
        (~predictions["symbol"].isin(held_symbols)) & (predictions["diff_ratio"] >= BUY_THRESHOLD)
    ].head(MAX_ORDERS_PER_RUN)

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
            _record_order(symbol, OrderSide.BUY, qty, result, broker, mode)
            logger.info(f"[exec] 買い発注: {symbol} {qty}株 @ 成行 (予測変化率={row['diff_ratio']:.3%})")
            stats["buy_orders"] += 1
        except Exception as e:
            logger.error(f"[exec] 買い注文エラー ({symbol}): {e}", exc_info=True)
            stats["errors"] += 1

    logger.info(
        f"=== 自動発注完了: 買い={stats['buy_orders']} 売り={stats['sell_orders']} "
        f"スキップ={stats['skipped']} エラー={stats['errors']} ==="
    )
    return stats
