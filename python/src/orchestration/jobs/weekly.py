"""週次スケジューラジョブ。

モデル学習・昇格・各種レポート・最適化・Walk-Forward 検証・
ウォッチリスト更新・DB メンテナンス・A/B 評価・ルール評価。
"""

from src.orchestration.jobs.common import _handle_stage_error, _is_first_week_of_month
from src.orchestration.types import PipelineStage
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
            logger.info(
                "昇格ゲート判定: eligible=%s reason=%s", gate_result.eligible, gate_result.reason
            )

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
            if _handle_stage_error(
                PipelineStage.CRITICAL, f"Challenger 学習 ({challenger_name})", e
            ):
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

        from src.prediction.db import (
            load_drift_summary,
            load_paper_real_diff_summary,
            save_weekly_accuracy_snapshot,
        )
        from src.reporting.discord.discord_utils import send_weekly_report

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
        from src.watchlist.batch_runner import load_target_symbols

        tasks = load_target_symbols()
        if use_optuna:
            from src.backtest.optimizer import run_optuna_batch

            logger.info("最適化エンジン: Optuna (n_trials=%d)", n_trials)
            results = run_optuna_batch(
                tasks,
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
                tasks,
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
        from src.watchlist.batch_runner import load_target_symbols

        tasks = load_target_symbols()
        result = run_walk_forward_comparison_report(
            tasks,
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
    週次 DB メンテナンス（土曜 03:00）: retention → CHECKPOINT/VACUUM を実行する。

    実行内容:
        1. retention  — 診断ログの古い行を削除（各グループ最新は保持）
        2. CHECKPOINT — WAL をメインファイルへフラッシュ
        3. VACUUM     — 削除済み領域を再利用可能化
        4. （月初週のみ）物理コンパクション — 再構築でファイル死領域を回収
    実行後にサイズ・所要時間を Discord 通知し、サイズが閾値超なら警告する。
    """
    import os
    import time
    from datetime import datetime, timezone

    from config.settings import DB_COMPACT_ENABLED, DB_LOG_RETENTION_DAYS, DB_SIZE_ALERT_GB
    from src.reporting.discord.discord_utils import (
        send_db_maintenance_completion,
        send_webhook_text,
    )
    from src.utils.data_path_utils import get_db_path
    from src.utils.db import _db_connection
    from src.utils.db.compact import compact_in_place
    from src.utils.db.retention import purge_old_training_logs

    logger.info("=== 週次 DB メンテナンス開始 ===")
    db_path = get_db_path()
    now = datetime.now(timezone.utc)

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
            # 1. retention: 診断ログの古い行を削除（肥大化の主因対策。各グループ最新は保持）
            deleted = purge_old_training_logs(con, DB_LOG_RETENTION_DAYS)
            total_deleted = sum(deleted.values())
            logger.info(
                "週次 DB メンテナンス: retention 削除 %d 行 (%s)",
                total_deleted,
                deleted,
            )
            # 2. CHECKPOINT + VACUUM（削除済み領域を再利用可能にする）
            con.execute("CHECKPOINT")
            con.execute("VACUUM")
        logger.info("週次 DB メンテナンス: retention + CHECKPOINT + VACUUM 完了")

        # 3. 月初週のみ物理コンパクション（VACUUM では縮まない死領域を再構築で回収）
        #    _db_connection の with を抜けて接続を閉じた後に実行する（FileLock は内部で再取得）。
        if DB_COMPACT_ENABLED and _is_first_week_of_month(now):
            logger.info("月初週のため物理コンパクションを実行します")
            counts = compact_in_place(db_path, DB_LOG_RETENTION_DAYS, keep_backup=False, now=now)
            logger.info("物理コンパクション完了: %d テーブル再構築", len(counts))
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
        # サイズ監視: 閾値超なら警告（コンパクション後でも超えるなら異常な増加）
        size_after_gb = size_after / 1024
        if error_msg is None and size_after_gb > DB_SIZE_ALERT_GB:
            send_webhook_text(
                f"⚠️ DB サイズ警告: {size_after_gb:,.1f} GB が閾値 {DB_SIZE_ALERT_GB:,.1f} GB を超えています。"
                f"\nretention 設定や肥大化テーブルを確認してください。"
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
