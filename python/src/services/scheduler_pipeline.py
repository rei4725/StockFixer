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
