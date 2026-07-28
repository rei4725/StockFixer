"""
run_daily_orders — 自動発注オーケストレーター

DuckDB の最新予測結果を読み込み、RiskManager のゲートチェックを通過した
銘柄に対して Broker 経由で成行注文を発注する。

取引フロー:
    1. prediction_results から当日予測 Top N シグナルを取得
    2. RiskManager.is_trading_allowed() で当日取引可否を確認
    3. 保有済みポジションを確認し重複買いを回避
    4. RiskManager.calc_position_size() で発注株数を算出
    5. broker.send_order() で成行注文（paper_orders への記録は broker 側の責務）
    6. paper_orders への predicted_at/signal_price/horizon 等の補完・paper_real_diff 記録
"""

import uuid

from config.settings import (
    CORRELATION_ENC_THRESHOLD,
    CORRELATION_WINDOW_DAYS,
    ENABLE_SHORT_SIDE,
    MAX_ORDERS_PER_RUN,
    MAX_POSITIONS,
    MIN_CHANGE_RATIO,
)
from src.domain.ports import (
    AlertLevel,
    MarketDataPort,
    NotificationPort,
    PredictionResultRepository,
)
from src.domain.trading_rules import get_lot_size
from src.trading.brokers.base import BrokerBase, BrokerError, OrderSide
from src.trading.correlation_risk import evaluate_correlation_gate
from src.trading.risk_manager import RiskManager
from src.trading.signal_generator import apply_multi_horizon_score_column
from src.trading.types import TradingGateStatus
from src.utils.db import save_order_run_summary
from src.utils.logger import get_logger

from .params import (
    _SPLIT_LOW_CONFIDENCE,
    _apply_split_qty,
    _attach_dynamic_thresholds,
    _calc_split_ratio,
    _choose_order_params,
    _resolve_kelly_params,
)
from .predictions import _load_latest_predictions
from .recording import _record_order, _sync_live_execution_diffs
from .selection import (
    _apply_buy_sector_limit,
    _compute_ml_exit_signals,
    _determine_entry_horizon,
    _get_held_symbols,
    _load_exit_model,
)
from .sl_tp import _check_sl_tp_exits
from .stats import OrderExecutionStats

logger = get_logger(__name__)


def run_daily_orders(
    broker: BrokerBase,
    market: str = "jp",
    mode: str = "paper",
    market_data: MarketDataPort | None = None,
    notifier: NotificationPort | None = None,
    prediction_repo: PredictionResultRepository | None = None,
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

    stats: OrderExecutionStats = {
        "buy_orders": 0,
        "sell_orders": 0,
        "short_orders": 0,
        "skipped": 0,
        "skipped_min_change": 0,
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
    }
    try:
        broker.get_token()
    except BrokerError as e:
        logger.error("[exec] トークン取得失敗。本日の発注をスキップします: %s", e, exc_info=True)
        if notifier is not None:
            notifier.send_alert(
                "kabu API トークンエラー",
                f"トークン取得に失敗したため本日の発注をスキップします。\n{e}",
                level=AlertLevel.ERROR,
            )
        return stats

    risk = RiskManager(broker, market=market)
    risk.update_peak_balance()  # R-307: DD基準値を発注前に更新
    lot = get_lot_size(market)

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

    if prediction_repo is not None:
        predictions = prediction_repo.get_latest_by_market(market)
    else:
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
    predictions = apply_multi_horizon_score_column(predictions)
    threshold_scale = float(predictions["threshold_scale"].iloc[0])
    logger.info(
        "[exec] 動的閾値適用: scale=%.3f buy_range=%.3f%%-%.3f%%",
        threshold_scale,
        predictions["effective_buy_threshold"].min() * 100,
        predictions["effective_buy_threshold"].max() * 100,
    )

    held_symbols = _get_held_symbols(broker)
    held_symbols_list = sorted(held_symbols)
    stats.update(
        {
            "daily_loss": gate_status.daily_loss,
            "daily_loss_limit": gate_status.daily_loss_limit,
            "n_held_symbols": len(held_symbols),
            "held_symbols_list": held_symbols_list,
        }
    )

    # --- 相関ゲート: 保有銘柄間の相関上昇時に新規買いをブロック ---
    corr_gate = evaluate_correlation_gate(
        held_symbols_list,
        market,
        window=CORRELATION_WINDOW_DAYS,
        enc_threshold=CORRELATION_ENC_THRESHOLD,
    )
    stats.update(
        {
            "correlation_blocked": not corr_gate.is_allowed,
            "enc": corr_gate.enc,
            "avg_correlation": corr_gate.avg_correlation,
        }
    )

    # --- SL/TP チェック: 含み損益が閾値を超えたポジションを強制クローズ ---
    sl_tp_triggered = _check_sl_tp_exits(broker, market, mode, market_data, stats)

    # --- 決済シグナル: 保有株で売りシグナルが出ているものを先にクローズ ---
    # ML エグジットモデルが利用可能な場合、モデルのシグナルも売り判定に加える
    exit_model = _load_exit_model(market)
    held_predictions = predictions[predictions["symbol"].isin(held_symbols)]
    ml_exit_symbols: set[str] = (
        _compute_ml_exit_signals(held_predictions, exit_model) if exit_model is not None else set()
    )

    sell_signals = predictions[
        (predictions["symbol"].isin(held_symbols))
        & (~predictions["symbol"].isin(sl_tp_triggered))
        & (
            (predictions["multi_horizon_score"] <= predictions["effective_sell_threshold"])
            | (predictions["symbol"].isin(ml_exit_symbols))
        )
    ]
    for _, row in sell_signals.iterrows():
        symbol = row["symbol"]
        # 予測変動量が閾値未満 → 発注スキップ（R-214）
        if abs(float(row.get("diff_ratio") or 0.0)) < MIN_CHANGE_RATIO:
            logger.info(
                "[exec] %s: diff_ratio=%.3f%% < min_change=%.3f%% → スキップ",
                symbol,
                float(row.get("diff_ratio") or 0) * 100,
                MIN_CHANGE_RATIO * 100,
            )
            stats["skipped_min_change"] += 1
            stats["skipped"] += 1
            continue
        try:
            # 保有全数を成行売り
            pos = next(
                (p for p in broker.get_positions() if p["symbol"].replace(".T", "") == symbol), None
            )
            if pos is None or pos["qty"] <= 0:
                continue
            qty = pos["qty"]
            order_type, order_price, order_reason, order_session = _choose_order_params(
                market=str(row["market"]),
                symbol=symbol,
                side=OrderSide.SELL,
                current_price=float(row.get("current_price") or 0.0),
                market_data=market_data,
            )
            result = broker.send_order(
                symbol,
                OrderSide.SELL,
                qty,
                price=order_price,
                order_type=order_type,
            )
            _record_order(
                market=str(row["market"]),
                predicted_at=str(row["predicted_at"]),
                symbol=symbol,
                side=OrderSide.SELL,
                qty=qty,
                signal_price=float(row.get("current_price") or 0.0),
                order_price=order_price,
                order_type=order_type,
                order_result=result,
                broker=broker,
                mode=mode,
                order_session=order_session,
            )
            logger.info(
                f"[exec] 売り発注: {symbol} {qty}株 @ {order_type.name}({order_reason}) "
                f"(1d変化率={row['diff_ratio']:.3%}, 統合スコア={row['multi_horizon_score']:.3%}, "
                f"閾値={row['effective_sell_threshold']:.3%})"
            )
            stats["sell_orders"] += 1
            stats["total_turnover"] += float(row.get("current_price") or 0.0) * qty
        except Exception as e:
            logger.error(f"[exec] 売り注文エラー ({symbol}): {e}", exc_info=True)
            stats["errors"] += 1

    # --- 新規買いシグナル ---
    if corr_gate.is_allowed:
        buy_candidates = predictions[
            (~predictions["symbol"].isin(held_symbols))
            & (predictions["multi_horizon_score"] >= predictions["effective_buy_threshold"])
        ]
        buy_signals = _apply_buy_sector_limit(buy_candidates).head(MAX_ORDERS_PER_RUN)
    else:
        logger.warning(
            "[exec] 相関ゲートにより新規買いをスキップ: %s",
            corr_gate.reason or "ENC閾値未満",
        )
        buy_signals = predictions.iloc[0:0]  # 空 DataFrame

    # 1ラン内で増える建玉数を追跡し、ラン途中でも MAX_POSITIONS を超えないようにする。
    # （ゲートはラン開始時の一度きりで、head(MAX_ORDERS_PER_RUN) だけでは保有上限を
    #   最大 MAX_ORDERS_PER_RUN-1 件超過しうるため）
    position_count = len(held_symbols)

    for _, row in buy_signals.iterrows():
        symbol = row["symbol"]
        current_price = float(row.get("current_price") or 0)

        if position_count >= MAX_POSITIONS:
            logger.info("[exec] 保有上限 %d 到達のため以降の新規買いをスキップ", MAX_POSITIONS)
            stats["skipped"] += 1
            continue

        # 予測変動量が閾値未満 → 発注スキップ（R-214）
        if abs(float(row.get("diff_ratio") or 0.0)) < MIN_CHANGE_RATIO:
            logger.info(
                "[exec] %s: diff_ratio=%.3f%% < min_change=%.3f%% → スキップ",
                symbol,
                float(row.get("diff_ratio") or 0) * 100,
                MIN_CHANGE_RATIO * 100,
            )
            stats["skipped_min_change"] += 1
            stats["skipped"] += 1
            continue

        if current_price <= 0:
            logger.warning(f"[exec] {symbol}: 現在値が取得できないためスキップ")
            stats["skipped"] += 1
            continue

        confidence = float(row.get("confidence_ratio") or 1.0)
        split_ratio = _calc_split_ratio(confidence)
        if split_ratio == 0.0:
            logger.info(
                "[exec] %s: confidence_ratio=%.3f < %.2f → 見送り (R-308)",
                symbol,
                confidence,
                _SPLIT_LOW_CONFIDENCE,
            )
            stats["skipped"] += 1
            continue

        kelly_win_rate, kelly_avg_win, kelly_avg_loss = _resolve_kelly_params(
            str(row["market"]), symbol
        )
        qty = risk.calc_position_size(
            symbol,
            current_price,
            win_rate=kelly_win_rate,
            avg_win=kelly_avg_win,
            avg_loss=kelly_avg_loss,
            confidence_ratio=confidence,
        )
        if qty <= 0:
            logger.info(f"[exec] {symbol}: 発注株数 0 → スキップ（残高不足or上限）")
            stats["skipped"] += 1
            continue

        if split_ratio < 1.0:
            qty = _apply_split_qty(qty, split_ratio, lot=lot)
            logger.info(
                "[exec] %s: confidence_ratio=%.3f → %.0f%%発注 %d株 (R-308)",
                symbol,
                confidence,
                split_ratio * 100,
                qty,
            )

        try:
            order_type, order_price, order_reason, order_session = _choose_order_params(
                market=str(row["market"]),
                symbol=symbol,
                side=OrderSide.BUY,
                current_price=current_price,
                market_data=market_data,
            )
            result = broker.send_order(
                symbol,
                OrderSide.BUY,
                qty,
                price=order_price,
                order_type=order_type,
            )
            _record_order(
                market=str(row["market"]),
                predicted_at=str(row["predicted_at"]),
                symbol=symbol,
                side=OrderSide.BUY,
                qty=qty,
                signal_price=current_price,
                order_price=order_price,
                order_type=order_type,
                order_result=result,
                broker=broker,
                mode=mode,
                order_session=order_session,
                split_ratio=split_ratio,
                horizon=_determine_entry_horizon(row),
            )
            logger.info(
                f"[exec] 買い発注: {symbol} {qty}株 @ {order_type.name}({order_reason}) "
                f"(1d変化率={row['diff_ratio']:.3%}, 統合スコア={row['multi_horizon_score']:.3%}, "
                f"閾値={row['effective_buy_threshold']:.3%}, sector={row.get('sector', 'N/A')}, "
                f"split={split_ratio:.0%})"
            )
            stats["buy_orders"] += 1
            stats["total_turnover"] += current_price * qty
            position_count += 1
        except Exception as e:
            logger.error(f"[exec] 買い注文エラー ({symbol}): {e}", exc_info=True)
            stats["errors"] += 1

    # --- ショートサイド処理（R-215）: ENABLE_SHORT_SIDE=True かつ PaperBroker のみ ---
    if ENABLE_SHORT_SIDE and broker.broker_name == "paper":
        # ショートポジション保有銘柄で買いシグナルが出ていれば SHORT_COVER
        try:
            short_positions = broker.get_short_positions()
        except AttributeError:
            short_positions = []
            logger.warning("[exec] get_short_positions() が未実装のため SHORT_COVER をスキップ")

        short_held_symbols = {p["symbol"] for p in short_positions if p.get("qty", 0) > 0}
        cover_signals = predictions[
            (predictions["symbol"].isin(short_held_symbols))
            & (predictions["multi_horizon_score"] >= predictions["effective_buy_threshold"])
        ]
        for _, row in cover_signals.iterrows():
            symbol = row["symbol"]
            try:
                pos = next((p for p in short_positions if p["symbol"] == symbol), None)
                if pos is None or pos["qty"] <= 0:
                    continue
                qty = pos["qty"]
                current_price = float(row.get("current_price") or 0.0)
                order_type, order_price, order_reason, order_session = _choose_order_params(
                    market=str(row["market"]),
                    symbol=symbol,
                    side=OrderSide.SHORT_COVER,
                    current_price=current_price,
                    market_data=market_data,
                )
                result = broker.send_order(
                    symbol,
                    OrderSide.SHORT_COVER,
                    qty,
                    price=order_price,
                    order_type=order_type,
                )
                _record_order(
                    market=str(row["market"]),
                    predicted_at=str(row["predicted_at"]),
                    symbol=symbol,
                    side=OrderSide.SHORT_COVER,
                    qty=qty,
                    signal_price=current_price,
                    order_price=order_price,
                    order_type=order_type,
                    order_result=result,
                    broker=broker,
                    mode=mode,
                    order_session=order_session,
                )
                logger.info(
                    f"[exec] SHORT_COVER発注: {symbol} {qty}株 @ {order_type.name}({order_reason}) "
                    f"(統合スコア={row['multi_horizon_score']:.3%})"
                )
                stats["short_orders"] += 1
                stats["total_turnover"] += current_price * qty
            except Exception as e:
                logger.error(f"[exec] SHORT_COVER注文エラー ({symbol}): {e}", exc_info=True)
                stats["errors"] += 1

        # 非保有かつ売りシグナルの銘柄に新規ショートエントリー
        short_entry_candidates = predictions[
            (~predictions["symbol"].isin(held_symbols))
            & (~predictions["symbol"].isin(short_held_symbols))
            & (predictions["multi_horizon_score"] <= predictions["effective_sell_threshold"])
        ].head(MAX_ORDERS_PER_RUN)

        for _, row in short_entry_candidates.iterrows():
            symbol = row["symbol"]
            current_price = float(row.get("current_price") or 0)

            if abs(float(row.get("diff_ratio") or 0.0)) < MIN_CHANGE_RATIO:
                logger.info(
                    "[exec] %s: diff_ratio=%.3f%% < min_change=%.3f%% → ショートスキップ",
                    symbol,
                    float(row.get("diff_ratio") or 0) * 100,
                    MIN_CHANGE_RATIO * 100,
                )
                stats["skipped_min_change"] += 1
                stats["skipped"] += 1
                continue

            if current_price <= 0:
                logger.warning(f"[exec] {symbol}: 現在値が取得できないためショートスキップ")
                stats["skipped"] += 1
                continue

            confidence = float(row.get("confidence_ratio") or 1.0)
            short_split_ratio = _calc_split_ratio(confidence)
            if short_split_ratio == 0.0:
                logger.info(
                    "[exec] %s: confidence_ratio=%.3f < %.2f → ショート見送り (R-308)",
                    symbol,
                    confidence,
                    _SPLIT_LOW_CONFIDENCE,
                )
                stats["skipped"] += 1
                continue

            kelly_win_rate, kelly_avg_win, kelly_avg_loss = _resolve_kelly_params(
                str(row["market"]), symbol
            )
            qty = risk.calc_position_size(
                symbol,
                current_price,
                win_rate=kelly_win_rate,
                avg_win=kelly_avg_win,
                avg_loss=kelly_avg_loss,
                confidence_ratio=confidence,
            )
            if qty <= 0:
                logger.info(f"[exec] {symbol}: 発注株数 0 → ショートスキップ（残高不足or上限）")
                stats["skipped"] += 1
                continue

            if short_split_ratio < 1.0:
                qty = _apply_split_qty(qty, short_split_ratio, lot=lot)
                logger.info(
                    "[exec] %s: confidence_ratio=%.3f → %.0f%%ショート発注 %d株 (R-308)",
                    symbol,
                    confidence,
                    short_split_ratio * 100,
                    qty,
                )

            try:
                order_type, order_price, order_reason, order_session = _choose_order_params(
                    market=str(row["market"]),
                    symbol=symbol,
                    side=OrderSide.SHORT,
                    current_price=current_price,
                    market_data=market_data,
                )
                result = broker.send_order(
                    symbol,
                    OrderSide.SHORT,
                    qty,
                    price=order_price,
                    order_type=order_type,
                )
                _record_order(
                    market=str(row["market"]),
                    predicted_at=str(row["predicted_at"]),
                    symbol=symbol,
                    side=OrderSide.SHORT,
                    qty=qty,
                    signal_price=current_price,
                    order_price=order_price,
                    order_type=order_type,
                    order_result=result,
                    broker=broker,
                    mode=mode,
                    order_session=order_session,
                    split_ratio=short_split_ratio,
                    horizon=_determine_entry_horizon(row),
                )
                logger.info(
                    f"[exec] ショート発注: {symbol} {qty}株 @ {order_type.name}({order_reason}) "
                    f"(1d変化率={row['diff_ratio']:.3%}, 統合スコア={row['multi_horizon_score']:.3%}, "
                    f"閾値={row['effective_sell_threshold']:.3%}, split={short_split_ratio:.0%})"
                )
                stats["short_orders"] += 1
                stats["total_turnover"] += current_price * qty
            except Exception as e:
                logger.error(f"[exec] ショート注文エラー ({symbol}): {e}", exc_info=True)
                stats["errors"] += 1

    logger.info(
        f"=== 自動発注完了: 買い={stats['buy_orders']} 売り={stats['sell_orders']} "
        f"ショート={stats['short_orders']} スキップ={stats['skipped']} エラー={stats['errors']} ==="
    )
    if mode == "live":
        _sync_live_execution_diffs(broker)
    # 発注サマリーを保存（R-214）
    _run_id = str(uuid.uuid4())[:12]
    try:
        save_order_run_summary(
            run_id=_run_id,
            market=market,
            mode=mode,
            buy_orders=stats["buy_orders"],
            sell_orders=stats["sell_orders"],
            short_orders=stats["short_orders"],
            skipped=stats["skipped"],
            skipped_min_change=stats["skipped_min_change"],
            total_turnover=stats["total_turnover"],
            min_change_ratio=MIN_CHANGE_RATIO,
        )
    except Exception:
        logger.error("[exec] order_run_summary 保存失敗", exc_info=True)
    return stats
