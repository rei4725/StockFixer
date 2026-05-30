"""
スケジューラパイプラインの業務ロジック

定期実行スケジューラから呼び出されるオーケストレーション層。
各パイプラインの制御フロー、エラーハンドリングをここで実装。
"""

from typing import Callable, Optional

from src.orchestration.types import PipelineStage
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _handle_stage_error(
    stage: PipelineStage,
    label: str,
    exc: Exception,
    notify_fn: Optional[Callable[[str], object]] = None,
) -> bool:
    """ステージ分類に基づくエラーハンドリング。

    Returns:
        True  → 呼び出し元は raise すべき (CRITICAL)
        False → 継続してよい (NON_CRITICAL / RECOVERABLE)
    """
    if stage is PipelineStage.CRITICAL:
        logger.error("%s 失敗: %s", label, exc, exc_info=True)
        if notify_fn is not None:
            notify_fn(f"{label} 失敗: {exc}")
        return True
    elif stage is PipelineStage.NON_CRITICAL:
        logger.error("%s 失敗: %s", label, exc, exc_info=True)
        return False
    else:  # RECOVERABLE
        logger.warning("%s 失敗（継続）: %s", label, exc, exc_info=True)
        return False


def run_daily_pipeline():
    """
    毎日実行: データ取得 → 予測 → Challenger shadow 予測 → 精度チェック → ドリフト監視 → Discord通知

    流れ:
        1. 全マーケットのデータを取得（バッチ）
        2. Top10/Worst10の予測を実行（production）
        2.5. Challenger shadow 予測（モデルが存在する場合のみ、非致命的）
        3. 前日予測の精度チェック: production / challenger それぞれ記録（非致命的）
        4. 日次ドリフトチェック（閾値超過銘柄を自動再学習）
        5. Discord通知
    """
    logger.info("=== 日次パイプライン開始 ===")

    from src.reporting.discord.discord_utils import (
        send_daily_pipeline_completion,
        send_daily_pipeline_error,
    )

    # 1. データ取得（CRITICAL: 失敗時はパイプライン停止 + Discord通知）
    logger.info("[1/5] データ取得開始")
    from src.market_data.pipeline import run_data_batch

    try:
        run_data_batch()
        logger.info("[1/5] データ取得完了")
    except Exception as e:
        if _handle_stage_error(PipelineStage.CRITICAL, "[1/5] データ取得", e, send_daily_pipeline_error):
            raise

    # 2. 予測（CRITICAL: 失敗時はパイプライン停止 + Discord通知）
    logger.info("[2/5] 予測開始 (production)")
    from src.prediction.prediction_pipeline import output_top_worst_results, predict_all_unified

    try:
        output_rows = predict_all_unified()
        output_top_worst_results(
            output_rows, mode="unified", shadow_mode=True, model_version="production"
        )
        logger.info("[2/5] 予測完了 (production): %d 銘柄", len(output_rows))
    except Exception as e:
        if _handle_stage_error(
            PipelineStage.CRITICAL,
            "[2/5] 予測 (production)",
            e,
            send_daily_pipeline_error,
        ):
            raise

    # 2.5. Challenger shadow 予測（NON_CRITICAL: モデルが存在しない場合はスキップ、失敗しても継続）
    logger.info("[2.5/5] Challenger shadow 予測開始")
    try:
        from src.prediction.shadow_evaluation import predict_with_challenger_unified

        challenger_rows = predict_with_challenger_unified()
        if challenger_rows:
            output_top_worst_results(
                challenger_rows, mode="unified", shadow_mode=True, model_version="challenger"
            )
            logger.info("[2.5/5] Challenger shadow 予測完了: %d 銘柄", len(challenger_rows))
        else:
            logger.info("[2.5/5] Challenger モデルなし（スキップ）")
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, "[2.5/5] Challenger shadow 予測", e)

    # 3. 前日予測の精度チェック（NON_CRITICAL: 失敗しても後続処理を継続）
    logger.info("[3/5] 予測精度チェック開始")
    try:
        from src.prediction.prediction_pipeline import run_accuracy_check
        from src.reporting.discord.discord_utils import send_accuracy_summary

        summary = run_accuracy_check(
            horizon=1, model_name="production", model_version_filter="production"
        )
        send_accuracy_summary(summary, horizon=1)
        logger.info("[3/5] 予測精度チェック完了 (production)")
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, "[3/5] 予測精度チェック (production)", e)

    try:
        from src.prediction.prediction_pipeline import run_accuracy_check

        run_accuracy_check(horizon=1, model_name="challenger", model_version_filter="challenger")
        logger.info("[3/5] 予測精度チェック完了 (challenger)")
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, "[3/5] 予測精度チェック (challenger)", e)

    # 4. 日次ドリフトチェック（NON_CRITICAL: 失敗しても後続処理を継続）
    logger.info("[4/5] 日次ドリフトチェック開始")
    try:
        run_daily_drift_check()
        logger.info("[4/5] 日次ドリフトチェック完了")
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, "[4/5] 日次ドリフトチェック", e)

    # 5. Discord通知（CRITICAL: 失敗時はパイプライン停止）
    logger.info("[5/5] Discord通知送信")
    try:
        send_daily_pipeline_completion()
        logger.info("[5/5] Discord通知完了")
    except Exception as e:
        if _handle_stage_error(PipelineStage.CRITICAL, "[5/5] Discord通知", e):
            raise

    logger.info("=== 日次パイプライン完了 ===")


def run_weekly_training():
    """
    週次実行: Shadow 評価 → 昇格ゲート → Challenger 再学習 → 精度チェック → Discord通知

    流れ:
        1. Shadow 評価: 前週の production vs challenger の Hit Rate / Sharpe を比較
        2. 昇格ゲート: 全基準（Net Return/MDD/Sharpe/Hit Rate/Slippage）を判定
        3. 昇格: challenger_wins かつ eligible なら Challenger → production に昇格
        4. Challenger 再学習（新しい challenger を学習して次週の評価に備える）
        5. 予測精度チェック & ドリフト警告
        6. Discord 通知（昇格結果含む）
    """
    logger.info("=== 週次モデル学習開始 ===")

    from config.settings import AUTO_PROMOTE_MODEL
    from src.prediction.promotion_gate import evaluate_promotion, save_promotion_result
    from src.prediction.shadow_evaluation import (
        _UNIFIED_CHALLENGER_NAMES,
        _UNIFIED_PRODUCTION_NAMES,
        evaluate_shadow_models,
        promote_unified_challenger,
    )
    from src.prediction.unified_model_pipeline import train_unified_model

    # 1. Shadow 評価（NON_CRITICAL: データなし時はスキップして継続）
    logger.info("[1/4] Shadow 評価開始")
    shadow_results = []
    for prod_name, chal_name in zip(_UNIFIED_PRODUCTION_NAMES, _UNIFIED_CHALLENGER_NAMES):
        try:
            result = evaluate_shadow_models(
                production_version="production",
                challenger_version="challenger",
            )
            shadow_results.append((prod_name, chal_name, result))
            logger.info(
                "Shadow 評価: production=hit=%s sharpe=%s / challenger=hit=%s sharpe=%s / wins=%s",
                result["production_hit_rate"],
                result["production_sharpe"],
                result["challenger_hit_rate"],
                result["challenger_sharpe"],
                result["challenger_wins"],
            )
            break  # prediction_accuracy の model_name は "production"/"challenger" で共通なので1回でよい
        except Exception as e:
            _handle_stage_error(PipelineStage.NON_CRITICAL, "Shadow 評価", e)

    # 2 & 3. 昇格ゲート + 昇格（RECOVERABLE: 昇格不可でも後続の再学習は継続）
    promoted = False
    gate_result = None
    if shadow_results and shadow_results[0][2]["challenger_wins"]:
        logger.info("[2/4] 昇格ゲート評価開始")
        try:
            gate_result = evaluate_promotion(
                shadow_model_name="challenger",
                current_model_name="production",
                require_manual_approval=not AUTO_PROMOTE_MODEL,
            )
            save_promotion_result(gate_result)
            logger.info("昇格ゲート判定: eligible=%s reason=%s", gate_result.eligible, gate_result.reason)

            if gate_result.eligible and AUTO_PROMOTE_MODEL:
                logger.info("[3/4] 昇格実行: challenger → production (AUTO_PROMOTE_MODEL=true)")
                promote_result = promote_unified_challenger()
                if promote_result["promoted"]:
                    logger.info("昇格完了: %s", promote_result["promoted"])
                    promoted = True
                else:
                    logger.warning("昇格対象ファイルなし: skipped=%s", promote_result["skipped"])
            elif gate_result.eligible:
                logger.info("[3/4] 昇格基準クリアだが AUTO_PROMOTE_MODEL=false のため手動承認待ち")
            else:
                logger.info("昇格ゲート未達（再学習のみ実施）: %s", gate_result.reason)
        except Exception as e:
            _handle_stage_error(PipelineStage.RECOVERABLE, "昇格ゲート評価", e)
    elif shadow_results:
        logger.info("[2/4] challenger_wins=False のため昇格ゲートをスキップ")
    else:
        logger.info("[2/4] Shadow 評価データなし（初回実行）のため昇格ゲートをスキップ")

    # 4. Challenger 再学習（CRITICAL: 失敗時はパイプライン停止）
    logger.info("[4/4] Challenger 再学習開始")
    for model_type, challenger_name in zip(
        ["XGBoostModel", "LightGBMModel"], _UNIFIED_CHALLENGER_NAMES
    ):
        try:
            logger.info("Challenger 学習開始: %s", challenger_name)
            train_unified_model(model_type=model_type, model_name=challenger_name)
            logger.info("Challenger 学習完了: %s", challenger_name)
        except Exception as e:
            if _handle_stage_error(PipelineStage.CRITICAL, f"Challenger 学習 ({challenger_name})", e):
                raise

    # 予測精度チェック & ドリフト警告（NON_CRITICAL: 失敗しても継続）
    logger.info("予測精度チェック開始")
    try:
        from src.prediction.prediction_pipeline import run_accuracy_check
        from src.reporting.discord.discord_utils import send_drift_alert

        summary = run_accuracy_check(
            horizon=1, model_name="production", model_version_filter="production"
        )
        send_drift_alert(summary, horizon=1)
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, "予測精度チェック", e)

    # Discord 完了通知（NON_CRITICAL: 通知失敗は警告ログのみ）
    try:
        from src.reporting.discord.discord_utils import (
            send_promotion_result,
            send_weekly_training_completion,
        )

        send_weekly_training_completion(_UNIFIED_CHALLENGER_NAMES)
        if gate_result is not None:
            send_promotion_result(
                promoted=promoted,
                reason=gate_result.reason,
                criteria=gate_result.criteria,
            )
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, "週次学習完了通知", e)

    logger.info("=== 週次モデル学習完了 ===")


def run_weekly_report():
    """
    週次パフォーマンスレポートを Discord に送信する。

    直近の予測精度サマリーを集計し、Discord Webhook へ通知することで
    モデルの劣化を早期検知する。
    """
    logger.info("=== 週次レポート生成開始 ===")
    try:
        from datetime import date, timedelta

        from src.reporting.discord.discord_utils import send_weekly_report
        from src.utils.db import (
            load_drift_summary,
            load_paper_real_diff_summary,
            save_weekly_accuracy_snapshot,
        )

        summary = load_drift_summary(horizon=1)

        # 今週月曜日を week_start として週次スナップショットを保存
        today = date.today()
        week_start = (today - timedelta(days=today.weekday())).isoformat()
        save_weekly_accuracy_snapshot(week_start, summary)

        diff_summary = load_paper_real_diff_summary(recent_days=7)
        send_weekly_report(accuracy_df=summary, horizon=1, diff_summary=diff_summary)
        logger.info("=== 週次レポート送信完了 ===")
    except Exception as e:
        logger.error("週次レポート生成失敗: %s", e, exc_info=True)

    # 外れ原因分析（非致命的）
    logger.info("外れ原因分析開始")
    try:
        from src.prediction.db import load_top_prediction_misses
        from src.prediction.miss_analysis import run_miss_analysis_batch
        from src.reporting.discord.discord_utils import send_miss_analysis_summary

        miss_df = load_top_prediction_misses(horizon=1, top_n=10, since_days=30)
        analysis_results = run_miss_analysis_batch(miss_df)
        send_miss_analysis_summary(miss_df, analysis_results, since_days=30)
        logger.info("外れ原因分析完了")
    except Exception as e:
        logger.error("外れ原因分析失敗: %s", e, exc_info=True)


def run_daily_auto_order():
    """
    毎営業日 8:50 実行: 前夜の予測シグナルに基づきペーパートレード注文を発注する。

    本番切り替え時は環境変数 AUTO_TRADE_MODE=live に変更する。
    """
    import os

    from src.trading.brokers.paper.paper_broker import PaperBroker
    from src.trading.execution import run_daily_orders

    mode = os.environ.get("AUTO_TRADE_MODE", "paper")
    logger.info("=== 自動発注開始 (mode=%s) ===", mode)

    if mode == "live":
        from src.trading.brokers.kabu.kabu_client import KabuBroker

        broker = KabuBroker()
    else:
        broker = PaperBroker()

    try:
        stats = run_daily_orders(broker=broker, market="jp", mode=mode)
        logger.info("=== 自動発注完了: 買い=%s 売り=%s ===", stats["buy_orders"], stats["sell_orders"])
    except Exception as e:
        logger.error("自動発注失敗: %s", e, exc_info=True)
        raise

    # Discord 完了通知
    try:
        from src.reporting.discord.discord_utils import send_daily_order_completion

        send_daily_order_completion(
            buy_orders=stats["buy_orders"],
            sell_orders=stats["sell_orders"],
            mode=mode,
            trading_stopped=stats.get("trading_stopped", False),
            stop_reason=stats.get("stop_reason"),
            daily_loss=stats.get("daily_loss"),
            daily_loss_limit=stats.get("daily_loss_limit"),
        )
    except Exception as e:
        logger.error("自動発注完了通知失敗: %s", e, exc_info=True)

    # ホライズン期限切れポジションの自動決済
    try:
        run_horizon_exit_check()
    except Exception as e:
        logger.error("ホライズン決済チェック失敗: %s", e, exc_info=True)

    # 相関リスク警告通知
    if stats.get("correlation_blocked"):
        try:
            from config.settings import CORRELATION_ENC_THRESHOLD
            from src.reporting.discord.discord_utils import send_correlation_alert

            send_correlation_alert(
                enc=stats.get("enc", 0.0),
                enc_threshold=CORRELATION_ENC_THRESHOLD,
                avg_correlation=stats.get("avg_correlation", 0.0),
                n_symbols=stats.get("n_held_symbols", 0),
                symbols=stats.get("held_symbols_list", []),
            )
        except Exception as e:
            logger.error("相関リスク警告通知失敗: %s", e, exc_info=True)


def run_horizon_exit_check() -> None:
    """
    毎営業日実行: target_exit_date を過ぎたペーパーポジションを成行売りで強制決済する。

    SL/TP によって既に決済済みのポジションは paper_positions に存在しないため
    自然に除外される（SL/TP 優先）。
    live モードでは処理をスキップする。
    """
    import os
    from datetime import date

    from src.trading.brokers.base import OrderSide
    from src.trading.brokers.paper.paper_broker import PaperBroker
    from src.utils.db._connection import _db_connection

    mode = os.environ.get("AUTO_TRADE_MODE", "paper")
    if mode == "live":
        logger.info("live モードのためホライズン決済チェックをスキップ")
        return

    today_str = date.today().isoformat()
    with _db_connection() as con:
        rows = con.execute(
            """
            SELECT DISTINCT symbol
            FROM paper_orders
            WHERE side = ?
              AND status = 'filled'
              AND target_exit_date IS NOT NULL
              AND CAST(target_exit_date AS VARCHAR) <= ?
            """,
            [int(OrderSide.BUY), today_str],
        ).fetchall()

    symbols_to_exit = [row[0] for row in rows]
    if not symbols_to_exit:
        logger.info("[horizon_exit] 期限切れポジションなし")
        return

    logger.info("[horizon_exit] 期限切れポジション対象: %s", symbols_to_exit)
    broker = PaperBroker()
    exited: list[str] = []
    for symbol in symbols_to_exit:
        positions = broker.get_positions()
        pos = next((p for p in positions if p["symbol"].replace(".T", "") == symbol), None)
        if pos is None or pos.get("qty", 0) <= 0:
            continue
        try:
            broker.send_order(symbol, OrderSide.SELL, pos["qty"])
            logger.info("[horizon_exit] ホライズン決済: %s %d株", symbol, pos["qty"])
            exited.append(symbol)
        except Exception as e:
            logger.error("[horizon_exit] 決済失敗: %s: %s", symbol, e, exc_info=True)

    logger.info("[horizon_exit] ホライズン決済完了: %d 件", len(exited))


def run_daily_settle_orders():
    """
    毎営業日 9:05 実行: pending 状態のペーパートレード注文を当日始値で約定処理する。
    本番（live）モードでは kabu STATION® が自動処理するため不要。
    """
    import os

    mode = os.environ.get("AUTO_TRADE_MODE", "paper")
    if mode == "live":
        logger.info("live モードのため settle スキップ")
        return

    from src.trading.brokers.paper.paper_broker import PaperBroker

    logger.info("=== ペーパートレード約定処理開始 ===")
    try:
        broker = PaperBroker()
        settled = broker.settle_pending_orders()
        logger.info("=== 約定処理完了: %s 件 ===", len(settled))
    except Exception as e:
        logger.error("約定処理失敗: %s", e, exc_info=True)
        raise

    # Discord 完了通知
    try:
        from src.reporting.discord.discord_utils import send_daily_settle_completion

        send_daily_settle_completion(settled_count=len(settled))
    except Exception as e:
        logger.error("約定処理完了通知失敗: %s", e, exc_info=True)


def run_daily_paper_trade_report():
    """
    毎営業日 15:30 実行: ペーパートレードのポジション・損益レポートを Discord に送信する。

    内容:
        - 現在のポジション一覧（銘柄・保有数・平均取得価格・現在値・含み損益）
        - 通算損益サマリー（実現損益・含み損益・合計損益）
    """
    import os

    mode = os.environ.get("AUTO_TRADE_MODE", "paper")
    if mode == "live":
        logger.info("live モードのためペーパートレードレポートをスキップ")
        return

    logger.info("=== ペーパートレード損益レポート送信開始 ===")
    try:
        from src.reporting.discord.discord_utils import send_paper_trade_position_report
        from src.trading.brokers.paper.paper_broker import PaperBroker

        broker = PaperBroker()
        positions = broker.get_positions()
        summary = broker.get_pnl_summary()
        send_paper_trade_position_report(positions, summary)
        logger.info("=== ペーパートレード損益レポート送信完了 ===")
    except Exception as e:
        logger.error("ペーパートレードレポート送信失敗: %s", e, exc_info=True)


def run_weekly_optimization():
    """
    週次実行: 全銘柄バックテスト最適化バッチ

    週次モデル学習（土曜03:00）完了後に実行し、最適パラメータを更新する。

    流れ:
        1. ウォッチリスト全銘柄のグリッドサーチ（並列数3）
        2. 最適パラメータを config/optimal_params.json に保存
    """
    import os

    logger.info("=== 週次バックテスト最適化開始 ===")

    use_optuna = os.getenv("USE_OPTUNA", "").strip().lower() in ("1", "true", "yes")
    n_trials = int(os.getenv("OPTUNA_N_TRIALS", "50"))

    success, failed = 0, 0
    try:
        if use_optuna:
            from src.backtest.optimizer import run_optuna_batch

            logger.info("最適化エンジン: Optuna (n_trials=%d)", n_trials)
            results = run_optuna_batch(
                model_type="XGBoostModel",
                ensemble=False,
                source="file",
                n_splits=5,
                n_trials=n_trials,
                max_workers=3,
                sort_by="sharpe_ratio",
            )
        else:
            from src.backtest.optimizer import run_optimize_batch

            logger.info("最適化エンジン: グリッドサーチ")
            results = run_optimize_batch(
                model_type="XGBoostModel",
                ensemble=False,
                source="file",
                n_splits=5,
                max_workers=3,
                sort_by="sharpe_ratio",
            )
        success = sum(1 for r in results if not r.get("error"))
        failed = len(results) - success
        logger.info("=== 週次バックテスト最適化完了: 成功=%s, 失敗=%s ===", success, failed)
    except Exception as e:
        logger.error("週次バックテスト最適化失敗: %s", e, exc_info=True)

    # Discord 完了通知
    try:
        from src.reporting.discord.discord_utils import send_optimization_completion

        send_optimization_completion(success=success, failed=failed)
    except Exception as e:
        logger.error("週次最適化完了通知失敗: %s", e, exc_info=True)


def run_weekly_walk_forward_report():
    """
    週次実行: Walk-Forward 比較レポート生成

    標準条件（source=file, n_splits=5）で全銘柄を再検証し、
    前回スナップショットとの差分レポートを保存する。
    """
    logger.info("=== 週次 Walk-Forward 比較レポート開始 ===")
    result: dict = {}
    try:
        from src.backtest.walk_forward_report import run_walk_forward_comparison_report

        result = run_walk_forward_comparison_report(
            model_type="XGBoostModel",
            source="file",
            n_splits=5,
            threshold=0.0,
            fee_rate=0.001,
            slippage=0.0,
        )
        logger.info(
            "=== 週次 Walk-Forward 比較レポート完了: " "success=%s failed=%s total=%s ===",
            result.get("success"),
            result.get("failed"),
            result.get("total"),
        )
    except Exception as e:
        logger.error("Walk-Forward 比較レポート生成失敗: %s", e, exc_info=True)

    # Discord 完了通知
    try:
        from src.reporting.discord.discord_utils import send_walk_forward_report_completion

        send_walk_forward_report_completion(result)
    except Exception as e:
        logger.error("Walk-Forward レポート通知失敗: %s", e, exc_info=True)


def run_weekly_watchlist_refresh():
    """
    週次実行: ウォッチリスト自動更新

    指数構成銘柄（S&P500 / 日経225）をWikipediaから取得し、
    watchlist.json を差分更新する。
    上場廃止確認済み銘柄を除外し、新規追加銘柄を組み込む。
    """
    logger.info("=== 週次ウォッチリスト更新開始 ===")
    try:
        from src.reporting.discord.discord_utils import send_watchlist_update_report
        from src.watchlist.manager import run_watchlist_refresh

        diffs = run_watchlist_refresh()
        send_watchlist_update_report(diffs)
        logger.info("=== 週次ウォッチリスト更新完了 ===")
    except Exception as e:
        logger.error("ウォッチリスト更新失敗: %s", e, exc_info=True)


def run_weekly_db_maintenance() -> None:
    """
    週次 DB メンテナンス（土曜 03:00）: CHECKPOINT → VACUUM を実行する。

    実行内容:
        1. CHECKPOINT — WAL をメインファイルへフラッシュ
        2. VACUUM     — 削除済み領域を回収してファイルサイズを縮小
    実行前後の DB ファイルサイズと所要時間を Discord に通知する。
    """
    import os
    import time

    from src.reporting.discord.discord_utils import send_db_maintenance_completion
    from src.utils.data_path_utils import get_db_path
    from src.utils.db import _db_connection

    logger.info("=== 週次 DB メンテナンス開始 ===")
    db_path = get_db_path()

    def _mb(path: str) -> float:
        try:
            return os.path.getsize(path) / (1024 * 1024)
        except OSError:
            return 0.0

    size_before = _mb(db_path)
    start = time.monotonic()
    error_msg = None

    try:
        with _db_connection() as con:
            con.execute("CHECKPOINT")
            con.execute("VACUUM")
        logger.info("週次 DB メンテナンス: CHECKPOINT + VACUUM 完了")
    except Exception as e:
        logger.error("週次 DB メンテナンス失敗: %s", e, exc_info=True)
        error_msg = str(e)

    elapsed = time.monotonic() - start
    size_after = _mb(db_path)
    logger.info(
        "=== 週次 DB メンテナンス完了: %.1f 秒, %.2f MB → %.2f MB ===",
        elapsed,
        size_before,
        size_after,
    )

    try:
        send_db_maintenance_completion(
            elapsed_seconds=elapsed,
            size_before_mb=size_before,
            size_after_mb=size_after,
            error=error_msg,
        )
    except Exception as e:
        logger.error("DB メンテナンス通知失敗: %s", e, exc_info=True)


def run_weekly_shadow_evaluation() -> None:
    """
    週次実行: A/Bテスト（シャドーモード）評価

    production / challenger 両モデルの Hit Rate / Sharpe を比較し、
    challenger_wins=True のとき Discord に昇格候補を通知する。

    手動承認後に promote_challenger_to_production() を実行することで
    challenger を本番モデルへ昇格させることができる。
    """
    logger.info("=== 週次 A/Bテスト評価開始 ===")
    result: dict = {}
    try:
        from src.prediction.shadow_evaluation import evaluate_shadow_models

        result = evaluate_shadow_models()
        logger.info(
            "A/Bテスト評価完了: challenger_wins=%s (prod_hit=%s, chal_hit=%s)",
            result["challenger_wins"],
            result["production_hit_rate"],
            result["challenger_hit_rate"],
        )
    except Exception as e:
        logger.error("A/Bテスト評価失敗: %s", e, exc_info=True)
        return

    try:
        from src.reporting.discord.discord_utils import send_shadow_evaluation_notification

        send_shadow_evaluation_notification(result)
    except Exception as e:
        logger.error("A/Bテスト評価通知失敗: %s", e, exc_info=True)

    logger.info("=== 週次 A/Bテスト評価完了 ===")


def run_daily_drift_check():
    """
    日次ドリフト監視: 直近 20 営業日の MAE / Hit Rate を監視し、閾値超過銘柄を自動再学習する。

    環境変数:
        DRIFT_MAE_THRESHOLD:      MAE 閾値（デフォルト 0.02 = 2%）
        DRIFT_HIT_RATE_THRESHOLD: Hit Rate 閾値（デフォルト 0.45 = 45%）

    完了条件:
        - 閾値超過銘柄が存在する場合に銘柄別モデルを再学習
        - 再学習トリガーを Discord に通知
    """
    import os

    from src.utils.db import load_drift_summary
    from src.utils.db.system_config import get_config_value

    _mae_default = os.environ.get("DRIFT_MAE_THRESHOLD", "0.02")
    _hr_default = os.environ.get("DRIFT_HIT_RATE_THRESHOLD", "0.45")
    mae_threshold = float(get_config_value("drift.mae_threshold", _mae_default))
    hit_rate_threshold = float(get_config_value("drift.hit_rate_threshold", _hr_default))

    logger.info(
        f"=== 日次ドリフトチェック開始 (MAE閾値={mae_threshold:.2%}, HitRate閾値={hit_rate_threshold:.0%}) ==="
    )

    summary = load_drift_summary(horizon=1, recent_n=20)
    if summary is None or summary.empty:
        logger.info("ドリフトチェック: 精度データなし（スキップ）")
        return

    triggered = summary[
        (summary["mean_abs_error"] >= mae_threshold)
        | (summary["direction_accuracy"] <= hit_rate_threshold)
    ]

    if triggered.empty:
        logger.info("ドリフトチェック: 閾値超過銘柄なし")
        return

    triggered_list = triggered[
        ["market", "symbol", "mean_abs_error", "direction_accuracy"]
    ].to_dict("records")
    logger.warning("ドリフト検知: %s 銀柄が閾値超過 → 自動再学習開始", len(triggered_list))

    # Discord 通知（再学習開始前）
    try:
        from src.reporting.discord.discord_utils import send_drift_retrain_notification

        send_drift_retrain_notification(triggered_list, mae_threshold, hit_rate_threshold)
    except Exception as e:
        logger.error("ドリフト通知失敗: %s", e, exc_info=True)

    # 銘柄別モデル再学習
    from src.prediction.training_pipeline import train_models_for_symbol_task
    from src.watchlist.batch_runner import load_target_symbols

    all_tasks = {(t.market, t.symbol): t for t in load_target_symbols()}
    success_count = 0
    retrained_symbols: list = []
    for sym in triggered_list:
        task = all_tasks.get((sym["market"], sym["symbol"]))
        if task is None:
            logger.warning("ドリフト再学習: タスクが見つかりません (%s/%s)", sym["market"], sym["symbol"])
            continue
        try:
            logger.info("ドリフト再学習開始: %s/%s", sym["market"], sym["symbol"])
            result = train_models_for_symbol_task(task)
            if result.get("status") == "success":
                success_count += 1
                retrained_symbols.append(sym)
            else:
                logger.warning(
                    f"ドリフト再学習スキップ/失敗 ({sym['market']}/{sym['symbol']}): "
                    f"{result.get('reason') or result.get('error') or result.get('status')}"
                )
        except Exception as e:
            logger.error("ドリフト再学習失敗 (%s/%s): %s", sym["market"], sym["symbol"], e, exc_info=True)

    logger.info("=== 日次ドリフトチェック完了: 再学習=%s/%s 件 ===", success_count, len(triggered_list))

    # 特徴量除外提案通知（再学習成功銘柄の最新 Permutation Importance を通知）
    if retrained_symbols:
        try:
            from src.reporting.discord.discord_utils import send_feature_suggestion_notification
            from src.utils.db import load_feature_exclusion_candidates

            feature_suggestions = []
            for sym in retrained_symbols:
                candidates = load_feature_exclusion_candidates(sym["market"], sym["symbol"])
                feature_suggestions.append(
                    {"market": sym["market"], "symbol": sym["symbol"], "candidates": candidates}
                )
            send_feature_suggestion_notification(feature_suggestions)
        except Exception as e:
            logger.error("特徴量除外提案通知失敗: %s", e, exc_info=True)


def run_weekly_rule_evaluation() -> None:
    """
    週次実行（日曜 02:00）: 全銘柄 × 全ルールをバックテストし
    「最優秀ルール」を DuckDB に保存する。

    判定基準: 勝率 50% 以上 AND 純利益プラス
    完了後 Discord に評価サマリーを通知する。
    """
    from config.execution import RULE_EVAL_END, RULE_EVAL_MARKET, RULE_EVAL_START

    logger.info("=== 週次ルール評価開始 ===")
    market = RULE_EVAL_MARKET
    backtest_start = RULE_EVAL_START
    backtest_end = RULE_EVAL_END

    try:
        from src.backtest.rule_selector import evaluate_all_symbols

        summary = evaluate_all_symbols(
            market=market,
            backtest_start=backtest_start,
            backtest_end=backtest_end,
        )
        logger.info(
            "=== 週次ルール評価完了: 有効=%s/%s 銘柄 ===",
            summary["effective"],
            summary["evaluated"],
        )
    except Exception as e:
        logger.error("週次ルール評価失敗: %s", e, exc_info=True)
        raise

    try:
        from src.reporting.discord.discord_utils import send_rule_evaluation_completion

        send_rule_evaluation_completion(
            evaluated=summary["evaluated"],
            effective=summary["effective"],
            skipped=summary["skipped"],
            market=market,
        )
    except Exception as e:
        logger.error("ルール評価通知失敗: %s", e, exc_info=True)


def run_pre_close_alert() -> None:
    """
    毎営業日 15:00 実行: 保有ポジションを直近予測と照合し
    利確/損切り推奨度を Discord に通知する（R-409）。
    """
    import os

    mode = os.environ.get("AUTO_TRADE_MODE", "paper")
    if mode == "live":
        logger.info("live モードのため引け前アラートをスキップ")
        return

    logger.info("=== 引け前ポジション再評価アラート開始 ===")
    try:
        from src.trading.pre_close_alert_service import get_pre_close_alerts

        lines = get_pre_close_alerts()
        logger.info("引け前アラート評価完了: %d行", len(lines))
    except Exception as e:
        logger.error("引け前アラート評価失敗: %s", e, exc_info=True)
        raise

    try:
        from src.reporting.discord.discord_notification_specs import PRE_CLOSE_ALERT
        from src.reporting.discord.discord_utils import send_webhook_notification

        send_webhook_notification(
            PRE_CLOSE_ALERT.title, "\n".join(lines), color=PRE_CLOSE_ALERT.color
        )
        logger.info("=== 引け前ポジション再評価アラート送信完了 ===")
    except Exception as e:
        logger.error("引け前アラート通知失敗: %s", e, exc_info=True)


def run_monthly_report_job() -> None:
    """
    月次実行（毎月1日 03:00）: 月次KPIレポートを生成・保存・Discord通知する。

    手順:
        1. Walk-Forward KPI を集計（Net Return / MDD / Sharpe）
        2. Hit Rate / Avg Slippage を集計
        3. paper/real 乖離・ドリフト状況を含む Markdown を results/monthly/ に保存
        4. Discord に通知
    """
    from config.settings import DRIFT_ALERT_THRESHOLD, DRIFT_ALERT_WEEKS
    from src.prediction.drift_monitor import check_weekly_hit_rate_drift
    from src.reporting.discord.discord_utils import send_monthly_report_notification
    from src.reporting.monthly import run_monthly_report, save_monthly_report_to_file

    def _drift_checker():
        return check_weekly_hit_rate_drift(weeks=DRIFT_ALERT_WEEKS, threshold=DRIFT_ALERT_THRESHOLD)

    logger.info("=== 月次レポート生成開始 ===")
    try:
        summary = run_monthly_report()
        report_path = save_monthly_report_to_file(summary, drift_checker=_drift_checker)
        logger.info("=== 月次レポート保存完了: %s ===", report_path)
    except Exception as e:
        logger.error("月次レポート生成失敗: %s", e, exc_info=True)
        raise

    try:
        send_monthly_report_notification(
            target_month=summary.target_month,
            net_return=summary.net_return,
            max_drawdown=summary.max_drawdown,
            sharpe_ratio=summary.sharpe_ratio,
            hit_rate=summary.hit_rate,
            avg_slippage=summary.avg_slippage,
            symbol_count=summary.symbol_count,
            report_path=report_path,
        )
    except Exception as e:
        logger.error("月次レポート通知失敗: %s", e, exc_info=True)

    logger.info("=== 月次レポート完了 ===")


def run_daily_backup() -> None:
    """
    毎日深夜実行: DuckDB をタイムスタンプ付きディレクトリへコピーし、最大5世代を保持する。

    手順:
        1. CHECKPOINT で WAL をメインファイルへフラッシュ
        2. data/backups/YYYYMMDD_HHMMSS/ へファイルコピー
        3. 5世代超過分を古い順に削除
        4. Discord に完了通知
    """
    from src.orchestration.backup_pipeline import run_db_backup
    from src.reporting.discord.discord_utils import send_backup_completion

    logger.info("=== 日次バックアップ開始 ===")
    result = run_db_backup()

    try:
        send_backup_completion(
            backup_path=result["backup_path"],
            size_mb=result["size_mb"],
            elapsed_seconds=result["elapsed_seconds"],
            pruned_count=result["pruned_count"],
            error=result["error"],
        )
    except Exception as e:
        logger.error("バックアップ通知失敗: %s", e, exc_info=True)

    if result["error"]:
        raise RuntimeError(f"バックアップ失敗: {result['error']}")

    logger.info("=== 日次バックアップ完了 ===")


def run_daily_rule_signals() -> None:
    """
    日次実行（平日 16:00）: 有効ルールを持つ銘柄に当日シグナルを適用し
    ペーパートレード注文を自動発行する。

    フロー:
        1. rule_best_by_symbol から有効銘柄を読み込む
        2. 各銘柄の最新データにルールを適用して Buy/Sell/Hold を判定
        3. ペーパートレード注文を PaperBroker 経由で発行
        4. Discord にシグナルサマリーを通知
    """
    import os

    logger.info("=== 日次ルールシグナルジョブ開始 ===")
    market = os.environ.get("RULE_EVAL_MARKET", "jp")

    try:
        from src.rule_engine.pipeline import run_rule_signal_pipeline
        from src.trading.rule_execution import execute_rule_paper_trades

        signals = run_rule_signal_pipeline(market=market)
        trade_stats = execute_rule_paper_trades(signals=signals, market=market)
        logger.info(
            "=== 日次ルールシグナル完了: BUY=%s SELL=%s ===",
            trade_stats["buy_orders"],
            trade_stats["sell_orders"],
        )
    except Exception as e:
        logger.error("日次ルールシグナル失敗: %s", e, exc_info=True)
        raise

    try:
        from src.reporting.discord.discord_utils import send_rule_daily_signals

        send_rule_daily_signals(
            signals=signals,
            market=market,
            buy_orders=trade_stats["buy_orders"],
            sell_orders=trade_stats["sell_orders"],
        )
    except Exception as e:
        logger.error("ルールシグナル通知失敗: %s", e, exc_info=True)
