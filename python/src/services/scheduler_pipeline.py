"""
スケジューラパイプラインの業務ロジック

定期実行スケジューラから呼び出されるオーケストレーション層。
各パイプラインの制御フロー、エラーハンドリングをここで実装。
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_daily_pipeline():
    """
    毎日実行: データ取得 → 予測 → Discord通知用CSV出力

    流れ:
        1. 全マーケットのデータを取得（バッチ）
        2. Top10/Worst10の予測を実行
        3. Discord通知用CSVを出力し、Botに通知
    """
    logger.info("=== 日次パイプライン開始 ===")

    from src.api.discord_utils import send_daily_pipeline_completion, send_daily_pipeline_error

    # 1. データ取得（バッチ）
    logger.info("[1/2] データ取得開始")
    from src.services.data_pipeline import run_data_batch

    try:
        run_data_batch()
        logger.info("[1/2] データ取得完了")
    except Exception as e:
        logger.error(f"[1/2] データ取得失敗: {e}")
        send_daily_pipeline_error(f"データ取得失敗: {e}")
        raise

    # 2. 予測（Top10/Worst10）
    logger.info("[2/2] 予測開始")
    from src.services.prediction_pipeline import output_top_worst_results, predict_all_unified

    try:
        output_rows = predict_all_unified()
        output_top_worst_results(output_rows, mode="unified")
        logger.info("[2/2] 予測完了")
    except Exception as e:
        logger.error(f"[2/2] 予測失敗: {e}")
        send_daily_pipeline_error(f"予測失敗: {e}")
        raise

    # 3. Discord通知
    logger.info("[3/3] Discord通知送信")
    try:
        send_daily_pipeline_completion()
        logger.info("[3/3] Discord通知完了")
    except Exception as e:
        logger.error(f"[3/3] Discord通知失敗: {e}")
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
            logger.info(f"学習開始: {model_name}")
            train_unified_model(model_type=model_type, model_name=model_name)
            logger.info(f"学習完了: {model_name}")
        except Exception as e:
            logger.error(f"学習失敗 ({model_name}): {e}")
            raise

    # 予測精度チェック & ドリフト警告
    logger.info("予測精度チェック開始")
    try:
        from src.api.discord_utils import send_drift_alert
        from src.services.prediction_pipeline import run_accuracy_check

        summary = run_accuracy_check(horizon=1)
        send_drift_alert(summary, horizon=1)
    except Exception as e:
        logger.error(f"予測精度チェック失敗: {e}", exc_info=True)

    # Discord 完了通知
    try:
        from src.api.discord_utils import send_weekly_training_completion

        trained_models = [
            "UnifiedStockXGBoost",
            "UnifiedStockLightGBM",
        ]
        send_weekly_training_completion(trained_models)
    except Exception as e:
        logger.error(f"週次学習完了通知失敗: {e}", exc_info=True)

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
        from src.utils.db import load_drift_summary

        summary = load_drift_summary(horizon=1)
        send_weekly_report(accuracy_df=summary, horizon=1)
        logger.info("=== 週次レポート送信完了 ===")
    except Exception as e:
        logger.error(f"週次レポート生成失敗: {e}", exc_info=True)
        raise


def run_daily_auto_order():
    """
    毎営業日 8:50 実行: 前夜の予測シグナルに基づきペーパートレード注文を発注する。

    本番切り替え時は環境変数 AUTO_TRADE_MODE=live に変更する。
    """
    import os

    from src.brokers.paper.paper_broker import PaperBroker
    from src.services.order_execution_pipeline import run_daily_orders

    mode = os.environ.get("AUTO_TRADE_MODE", "paper")
    logger.info(f"=== 自動発注開始 (mode={mode}) ===")

    if mode == "live":
        from src.brokers.kabu.kabu_client import KabuBroker

        broker = KabuBroker()
    else:
        broker = PaperBroker()

    try:
        stats = run_daily_orders(broker=broker, market="jp", mode=mode)
        logger.info(f"=== 自動発注完了: 買い={stats['buy_orders']} 売り={stats['sell_orders']} ===")
    except Exception as e:
        logger.error(f"自動発注失敗: {e}", exc_info=True)
        raise

    # Discord 完了通知
    try:
        from src.api.discord_utils import send_daily_order_completion

        send_daily_order_completion(
            buy_orders=stats["buy_orders"],
            sell_orders=stats["sell_orders"],
            mode=mode,
        )
    except Exception as e:
        logger.error(f"自動発注完了通知失敗: {e}", exc_info=True)


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
        logger.info(f"=== 約定処理完了: {len(settled)} 件 ===")
    except Exception as e:
        logger.error(f"約定処理失敗: {e}", exc_info=True)
        raise

    # Discord 完了通知
    try:
        from src.api.discord_utils import send_daily_settle_completion

        send_daily_settle_completion(settled_count=len(settled))
    except Exception as e:
        logger.error(f"約定処理完了通知失敗: {e}", exc_info=True)


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
        logger.error(f"ペーパートレードレポート送信失敗: {e}", exc_info=True)


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
        logger.info(f"=== 週次バックテスト最適化完了: 成功={success}, 失敗={failed} ===")
    except Exception as e:
        logger.error(f"週次バックテスト最適化失敗: {e}", exc_info=True)
        raise

    # Discord 完了通知
    try:
        from src.api.discord_utils import send_optimization_completion

        send_optimization_completion(success=success, failed=failed)
    except Exception as e:
        logger.error(f"週次最適化完了通知失敗: {e}", exc_info=True)


def run_weekly_walk_forward_report():
    """
    週次実行: Walk-Forward 比較レポート生成

    標準条件（source=file, n_splits=5）で全銘柄を再検証し、
    前回スナップショットとの差分レポートを保存する。
    """
    logger.info("=== 週次 Walk-Forward 比較レポート開始 ===")
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
            "=== 週次 Walk-Forward 比較レポート完了: "
            f"success={result['success']} failed={result['failed']} total={result['total']} ==="
        )
    except Exception as e:
        logger.error(f"Walk-Forward 比較レポート生成失敗: {e}", exc_info=True)
        raise

    # Discord 完了通知
    try:
        from src.api.discord_utils import send_walk_forward_report_completion

        send_walk_forward_report_completion(result)
    except Exception as e:
        logger.error(f"Walk-Forward レポート通知失敗: {e}", exc_info=True)


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
        logger.error(f"ウォッチリスト更新失敗: {e}", exc_info=True)
        raise
