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

    from src.api.discord_utils import send_daily_pipeline_completion, send_daily_pipeline_error

    # 1. データ取得（バッチ）
    logger.info("[1/4] データ取得開始")
    from src.services.data_pipeline import run_data_batch

    try:
        run_data_batch()
        logger.info("[1/4] データ取得完了")
    except Exception as e:
        logger.error("[1/4] データ取得失敗: %s", e, exc_info=True)
        send_daily_pipeline_error(f"データ取得失敗: {e}")
        raise

    # 2. 予測（Top10/Worst10）
    logger.info("[2/4] 予測開始")
    from src.services.prediction_pipeline import output_top_worst_results, predict_all_unified

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
        from src.services.prediction_pipeline import run_accuracy_check

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

    from src.services.unified_model_pipeline import train_unified_model

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
        from src.api.discord_utils import send_drift_alert
        from src.services.prediction_pipeline import run_accuracy_check

        summary = run_accuracy_check(horizon=1)
        send_drift_alert(summary, horizon=1)
    except Exception as e:
        logger.error("予測精度チェック失敗: %s", e, exc_info=True)

    # Discord 完了通知
    try:
        from src.api.discord_utils import send_weekly_training_completion

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
        from src.api.discord_utils import send_weekly_report
        from src.utils.db import load_drift_summary, load_paper_real_diff_summary

        summary = load_drift_summary(horizon=1)
        diff_summary = load_paper_real_diff_summary(recent_days=7)
        send_weekly_report(accuracy_df=summary, horizon=1, diff_summary=diff_summary)
        logger.info("=== 週次レポート送信完了 ===")
    except Exception as e:
        logger.error("週次レポート生成失敗: %s", e, exc_info=True)


def run_daily_auto_order():
    """
    毎営業日 8:50 実行: 前夜の予測シグナルに基づきペーパートレード注文を発注する。

    本番切り替え時は環境変数 AUTO_TRADE_MODE=live に変更する。
    """
    import os

    from src.brokers.paper.paper_broker import PaperBroker
    from src.services.order_execution_pipeline import run_daily_orders

    mode = os.environ.get("AUTO_TRADE_MODE", "paper")
    logger.info("=== 自動発注開始 (mode=%s) ===", mode)

    if mode == "live":
        from src.brokers.kabu.kabu_client import KabuBroker

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
        from src.api.discord_utils import send_daily_order_completion

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

    from src.brokers.paper.paper_broker import PaperBroker

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
        from src.api.discord_utils import send_daily_settle_completion

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
        from src.api.discord_utils import send_paper_trade_position_report
        from src.brokers.paper.paper_broker import PaperBroker

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
    logger.info("=== 週次バックテスト最適化開始 ===")

    from src.services.backtest_optimize_pipeline import run_optimize_batch

    success, failed = 0, 0
    try:
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
        from src.api.discord_utils import send_optimization_completion

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
        from src.services.walk_forward_report_pipeline import run_walk_forward_comparison_report

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
        from src.api.discord_utils import send_walk_forward_report_completion

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
        from src.api.discord_utils import send_watchlist_update_report
        from src.services.watchlist_manager import run_watchlist_refresh

        diffs = run_watchlist_refresh()
        send_watchlist_update_report(diffs)
        logger.info("=== 週次ウォッチリスト更新完了 ===")
    except Exception as e:
        logger.error("ウォッチリスト更新失敗: %s", e, exc_info=True)


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

    mae_threshold = float(os.environ.get("DRIFT_MAE_THRESHOLD", "0.02"))
    hit_rate_threshold = float(os.environ.get("DRIFT_HIT_RATE_THRESHOLD", "0.45"))

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
        from src.api.discord_utils import send_drift_retrain_notification

        send_drift_retrain_notification(triggered_list, mae_threshold, hit_rate_threshold)
    except Exception as e:
        logger.error("ドリフト通知失敗: %s", e, exc_info=True)

    # 銘柄別モデル再学習
    from src.services.batch_runner import load_target_symbols
    from src.services.model_training_pipeline import train_models_for_symbol_task

    all_tasks = {(t.market, t.symbol): t for t in load_target_symbols()}
    success_count = 0
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
            else:
                logger.warning(
                    f"ドリフト再学習スキップ/失敗 ({sym['market']}/{sym['symbol']}): "
                    f"{result.get('reason') or result.get('error') or result.get('status')}"
                )
        except Exception as e:
            logger.error("ドリフト再学習失敗 (%s/%s): %s", sym["market"], sym["symbol"], e, exc_info=True)

    logger.info("=== 日次ドリフトチェック完了: 再学習=%s/%s 件 ===", success_count, len(triggered_list))
