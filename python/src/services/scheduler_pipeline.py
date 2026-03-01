"""
スケジューラパイプラインの業務ロジック

定期実行スケジューラから呼び出されるオーケストレーション層。
各パイプラインの制御フロー、エラーハンドリングをここで実装。
"""

import logging

logger = logging.getLogger("scheduler")


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
    from src.services.prediction_pipeline import predict_all_unified, output_top_worst_results
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
    週次実行: 統合モデル再学習
    
    流れ:
        1. XGBoostモデルの再学習
        2. LightGBMモデルの再学習
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

    logger.info("=== 週次モデル学習完了 ===")
