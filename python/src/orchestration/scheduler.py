"""
スケジューラパイプラインの業務ロジック

定期実行スケジューラから呼び出されるオーケストレーション層。
各パイプラインの制御フロー、エラーハンドリングをここで実装。
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_daily_pipeline():
    """
    毎日実行: データ取得 → 予測 → 精度チェック → ドリフト監視 → Discord通知用CSV出力

    流れ:
        1. 全マーケットのデータを取得（バッチ）
        2. Top10/Worst10の予測を実行
        3. 前日予測の精度チェック（prediction_accuracy テーブルへ記録）
        4. 日次ドリフトチェック（閾値超過銘柄を自動再学習）
        5. Discord通知
    """
    logger.info("=== 日次パイプライン開始 ===")

    from src.reporting.discord.discord_utils import (
        send_daily_pipeline_completion,
        send_daily_pipeline_error,
    )

    # 1. データ取得（バッチ）
    logger.info("[1/4] データ取得開始")
    from src.market_data.pipeline import run_data_batch

    try:
        run_data_batch()
        logger.info("[1/4] データ取得完了")
    except Exception as e:
        logger.error("[1/4] データ取得失敗: %s", e, exc_info=True)
        send_daily_pipeline_error(f"データ取得失敗: {e}")
        raise

    # 2. 予測（Top10/Worst10）
    logger.info("[2/4] 予測開始")
    from src.prediction.prediction_pipeline import output_top_worst_results, predict_all_unified

    try:
        output_rows = predict_all_unified()
        output_top_worst_results(output_rows, mode="unified")
        logger.info("[2/4] 予測完了")
    except Exception as e:
        logger.error("[2/4] 予測失敗: %s", e, exc_info=True)
        send_daily_pipeline_error(f"予測失敗: {e}")
        raise

    # 3. 前日予測の精度チェック（非致命的：失敗しても後続処理を継続）
    logger.info("[3/4] 予測精度チェック開始")
    try:
        from src.prediction.prediction_pipeline import run_accuracy_check

        run_accuracy_check(horizon=1)
        logger.info("[3/4] 予測精度チェック完了")
    except Exception as e:
        logger.error("[3/4] 予測精度チェック失敗: %s", e, exc_info=True)

    # 4. 日次ドリフトチェック（非致命的：失敗しても後続処理を継続）
    logger.info("[4/4] 日次ドリフトチェック開始")
    try:
        run_daily_drift_check()
        logger.info("[4/4] 日次ドリフトチェック完了")
    except Exception as e:
        logger.error("[4/4] 日次ドリフトチェック失敗: %s", e, exc_info=True)

    # 5. Discord通知
    logger.info("[5/5] Discord通知送信")
    try:
        send_daily_pipeline_completion()
        logger.info("[5/5] Discord通知完了")
    except Exception as e:
        logger.error("[5/5] Discord通知失敗: %s", e, exc_info=True)
        raise

    logger.info("=== 日次パイプライン完了 ===")


def run_weekly_training():
    """
    週次実行: 統合モデル再学習 → 予測精度チェック → ドリフト警告

    流れ:
        1. XGBoostモデルの再学習
        2. LightGBMモデルの再学習
        3. 予測精度チェック & ドリフト警告
    """
    logger.info("=== 週次モデル学習開始 ===")

    from src.prediction.unified_model_pipeline import train_unified_model

    for model_type in ["XGBoostModel", "LightGBMModel"]:
        model_name = f"UnifiedStock{model_type.replace('Model', '')}"
        try:
            logger.info("学習開始: %s", model_name)
            train_unified_model(model_type=model_type, model_name=model_name)
            logger.info("学習完了: %s", model_name)
        except Exception as e:
            logger.error("学習失敗 (%s): %s", model_name, e, exc_info=True)
            raise

    # 予測精度チェック & ドリフト警告
    logger.info("予測精度チェック開始")
    try:
        from src.prediction.prediction_pipeline import run_accuracy_check
        from src.reporting.discord.discord_utils import send_drift_alert

        summary = run_accuracy_check(horizon=1)
        send_drift_alert(summary, horizon=1)
    except Exception as e:
        logger.error("予測精度チェック失敗: %s", e, exc_info=True)

    # Discord 完了通知
    try:
        from src.reporting.discord.discord_utils import send_weekly_training_completion

        trained_models = [
            "UnifiedStockXGBoost",
            "UnifiedStockLightGBM",
        ]
        send_weekly_training_completion(trained_models)
    except Exception as e:
        logger.error("週次学習完了通知失敗: %s", e, exc_info=True)

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
        from src.utils.db import load_drift_summary, load_paper_real_diff_summary
        from src.utils.db import save_weekly_accuracy_snapshot

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
        from src.prediction.miss_analysis import run_miss_analysis_batch
        from src.reporting.discord.discord_utils import send_miss_analysis_summary
        from src.utils.db import load_top_prediction_misses

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
        from src.trading.brokers.paper.paper_broker import PaperBroker
        from src.reporting.discord.discord_utils import send_paper_trade_position_report

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
    all_shap_results: list = []
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
                all_shap_results.extend(result.get("shap_results", []))
            else:
                logger.warning(
                    f"ドリフト再学習スキップ/失敗 ({sym['market']}/{sym['symbol']}): "
                    f"{result.get('reason') or result.get('error') or result.get('status')}"
                )
        except Exception as e:
            logger.error("ドリフト再学習失敗 (%s/%s): %s", sym["market"], sym["symbol"], e, exc_info=True)

    logger.info("=== 日次ドリフトチェック完了: 再学習=%s/%s 件 ===", success_count, len(triggered_list))

    # SHAP サマリーをまとめて 1 通送信
    if all_shap_results:
        try:
            from src.reporting.discord.discord_utils import send_shap_batch_summary

            send_shap_batch_summary(all_shap_results)
        except Exception as e:
            logger.error("SHAP サマリー通知失敗: %s", e, exc_info=True)


def run_weekly_rule_evaluation() -> None:
    """
    週次実行（日曜 02:00）: 全銘柄 × 全ルールをバックテストし
    「最優秀ルール」を DuckDB に保存する。

    判定基準: 勝率 50% 以上 AND 純利益プラス
    完了後 Discord に評価サマリーを通知する。
    """
    import os

    logger.info("=== 週次ルール評価開始 ===")
    market = os.environ.get("RULE_EVAL_MARKET", "jp")
    backtest_start = os.environ.get("RULE_EVAL_START", "2022-01-01")
    backtest_end = os.environ.get("RULE_EVAL_END", "2025-01-01")

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
        from src.trading.pre_close_alert_service import evaluate_positions

        alerts = evaluate_positions()
        logger.info("引け前アラート評価完了: %d件", len(alerts))
    except Exception as e:
        logger.error("引け前アラート評価失敗: %s", e, exc_info=True)
        raise

    try:
        from src.reporting.discord.discord_utils import send_pre_close_alert

        send_pre_close_alert(alerts)
        logger.info("=== 引け前ポジション再評価アラート送信完了 ===")
    except Exception as e:
        logger.error("引け前アラート通知失敗: %s", e, exc_info=True)


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
        from src.prediction.rule_signal_pipeline import (
            execute_rule_paper_trades,
            run_rule_signal_pipeline,
        )

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
