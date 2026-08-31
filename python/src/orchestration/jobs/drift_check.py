"""日次ドリフト監視ジョブ。

daily.py からの責務分離（ファイル行数ガード対策）。run_daily_pipeline() の
step4、および手動実行（run_scheduler.py --run-now drift）の両方から呼ばれる。
"""

from src.orchestration.jobs.common import _drift_check_lock, skip_if_running
from src.utils.logger import get_logger

logger = get_logger(__name__)


@skip_if_running(_drift_check_lock, "日次ドリフトチェック")
def run_daily_drift_check():
    """
    日次ドリフト監視: 直近 20 営業日の MAE / Hit Rate を監視し、閾値超過銘柄を自動再学習する。

    環境変数:
        DRIFT_MAE_THRESHOLD:       MAE 閾値（デフォルト 0.02 = 2%）
        DRIFT_HIT_RATE_THRESHOLD:  Hit Rate 閾値（デフォルト 0.45 = 45%）
        DRIFT_MIN_SAMPLES:         判定に必要な最小サンプル数（デフォルト 10）。
                                   これ未満の銘柄はサンプル不足のノイズとみなし対象外にする。
        DRIFT_MAX_RETRAIN_PER_RUN: 1回の実行で再学習する最大銘柄数（デフォルト 50）。
                                   超過分は MAE が悪い順に足切りし、翌営業日以降の判定に委ねる。

    完了条件:
        - 閾値超過銘柄が存在する場合に銘柄別モデルを再学習
        - 再学習トリガーを Discord に通知
    """
    import os

    from src.prediction.db import load_drift_summary
    from src.utils.db.system_config import get_config_value

    _mae_default = os.environ.get("DRIFT_MAE_THRESHOLD", "0.02")
    _hr_default = os.environ.get("DRIFT_HIT_RATE_THRESHOLD", "0.45")
    _min_samples_default = os.environ.get("DRIFT_MIN_SAMPLES", "10")
    _max_retrain_default = os.environ.get("DRIFT_MAX_RETRAIN_PER_RUN", "50")
    mae_threshold = float(get_config_value("drift.mae_threshold", _mae_default))
    hit_rate_threshold = float(get_config_value("drift.hit_rate_threshold", _hr_default))
    min_samples = int(get_config_value("drift.min_samples", _min_samples_default))
    max_retrain_per_run = int(get_config_value("drift.max_retrain_per_run", _max_retrain_default))

    logger.info(
        f"=== 日次ドリフトチェック開始 (MAE閾値={mae_threshold:.2%}, HitRate閾値={hit_rate_threshold:.0%}, "
        f"最小サンプル数={min_samples}, 再学習上限={max_retrain_per_run}) ==="
    )

    summary = load_drift_summary(horizon=1, recent_n=20)
    if summary is None or summary.empty:
        logger.info("ドリフトチェック: 精度データなし（スキップ）")
        return

    triggered = summary[
        (summary["mean_abs_error"] >= mae_threshold)
        | (summary["direction_accuracy"] <= hit_rate_threshold)
    ]
    triggered = triggered[triggered["n_samples"] >= min_samples]

    if triggered.empty:
        logger.info("ドリフトチェック: 閾値超過銘柄なし")
        return

    if len(triggered) > max_retrain_per_run:
        logger.warning(
            "ドリフト検知: %s 銘柄が閾値超過 → MAEが悪い順に上位 %s 件のみ再学習",
            len(triggered),
            max_retrain_per_run,
        )
        triggered = triggered.sort_values("mean_abs_error", ascending=False).head(
            max_retrain_per_run
        )

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
            logger.warning(
                "ドリフト再学習: タスクが見つかりません (%s/%s)", sym["market"], sym["symbol"]
            )
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
            logger.error(
                "ドリフト再学習失敗 (%s/%s): %s", sym["market"], sym["symbol"], e, exc_info=True
            )

    logger.info(
        "=== 日次ドリフトチェック完了: 再学習=%s/%s 件 ===", success_count, len(triggered_list)
    )

    # 特徴量除外提案通知（再学習成功銘柄の最新 Permutation Importance を通知）
    if retrained_symbols:
        try:
            from src.prediction.db import load_feature_exclusion_candidates
            from src.reporting.discord.discord_utils import send_feature_suggestion_notification

            feature_suggestions = []
            for sym in retrained_symbols:
                candidates = load_feature_exclusion_candidates(sym["market"], sym["symbol"])
                feature_suggestions.append(
                    {"market": sym["market"], "symbol": sym["symbol"], "candidates": candidates}
                )
            send_feature_suggestion_notification(feature_suggestions)
        except Exception as e:
            logger.error("特徴量除外提案通知失敗: %s", e, exc_info=True)
