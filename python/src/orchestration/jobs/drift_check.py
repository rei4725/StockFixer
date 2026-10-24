"""日次ドリフト監視ジョブ。

daily.py からの責務分離（ファイル行数ガード対策）。run_daily_pipeline() の
step4、および手動実行（run_scheduler.py --run-now drift）の両方から呼ばれる。
"""

from src.orchestration.jobs.alert_streaks import record_drift_warn_streak
from src.orchestration.jobs.common import _drift_check_lock, skip_if_running
from src.utils.logger import get_logger

logger = get_logger(__name__)

# MAE の実効の線 = max(DRIFT_MAE_THRESHOLD, 全銘柄の MAE の中央値 × この倍率)
DRIFT_MAE_RELATIVE_RATIO_DEFAULT = "1.5"
# Hit Rate の実効の線 = min(DRIFT_HIT_RATE_THRESHOLD, 全銘柄の Hit Rate の中央値 - この幅)
# 20件の Hit Rate の標本のぶれ（p=0.45 で標準偏差 約0.11）の約2倍を幅にとる（#1045）
DRIFT_HIT_RATE_RELATIVE_MARGIN_DEFAULT = "0.20"


@skip_if_running(_drift_check_lock, "日次ドリフトチェック")
def run_daily_drift_check():
    """
    日次ドリフト監視: 直近 20 営業日の MAE / Hit Rate を監視し、閾値超過銘柄を自動再学習する。

    環境変数:
        DRIFT_MAE_THRESHOLD:       MAE 閾値の下限（デフォルト 0.02 = 2%）
        DRIFT_MAE_RELATIVE_RATIO:  全銘柄の MAE の中央値に掛ける倍率（デフォルト 1.5）。
                                   実効の線は下限と「中央値×倍率」の大きい方。
        DRIFT_HIT_RATE_THRESHOLD:  Hit Rate 閾値の上限（デフォルト 0.45 = 45%）
        DRIFT_HIT_RATE_RELATIVE_MARGIN: 全銘柄の Hit Rate の中央値から引く幅（デフォルト 0.20）。
                                   実効の線は上限と「中央値-幅」の小さい方。
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
    _mae_ratio_default = os.environ.get(
        "DRIFT_MAE_RELATIVE_RATIO", DRIFT_MAE_RELATIVE_RATIO_DEFAULT
    )
    mae_threshold = float(get_config_value("drift.mae_threshold", _mae_default))
    hit_rate_threshold = float(get_config_value("drift.hit_rate_threshold", _hr_default))
    min_samples = int(get_config_value("drift.min_samples", _min_samples_default))
    max_retrain_per_run = int(get_config_value("drift.max_retrain_per_run", _max_retrain_default))
    mae_relative_ratio = float(get_config_value("drift.mae_relative_ratio", _mae_ratio_default))
    _hr_margin_default = os.environ.get(
        "DRIFT_HIT_RATE_RELATIVE_MARGIN", DRIFT_HIT_RATE_RELATIVE_MARGIN_DEFAULT
    )
    hit_rate_relative_margin = float(
        get_config_value("drift.hit_rate_relative_margin", _hr_margin_default)
    )

    logger.info(
        f"=== 日次ドリフトチェック開始 (MAE閾値={mae_threshold:.2%}, HitRate閾値={hit_rate_threshold:.0%}, "
        f"最小サンプル数={min_samples}, 再学習上限={max_retrain_per_run}) ==="
    )

    summary = load_drift_summary(horizon=1, recent_n=20, model_name="production")
    if summary is None or summary.empty:
        logger.info("ドリフトチェック: 精度データなし（スキップ）")
        return

    eligible = summary[summary["n_samples"] >= min_samples]
    # 全銘柄の MAE が同じ水準で高いときに全件が超えないよう、線を全銘柄の中央値の倍率まで上げる（#1020）
    if not eligible.empty:
        mae_threshold = max(
            mae_threshold, float(eligible["mean_abs_error"].median()) * mae_relative_ratio
        )
        logger.info(
            "ドリフトチェック: 実効MAE閾値=%.2f%% (中央値×%s と下限の大きい方)",
            mae_threshold * 100,
            mae_relative_ratio,
        )
        # 全銘柄の Hit Rate が同じ水準で低いときに半数が超えないよう、線を中央値から幅を引いた値まで下げる（#1045）
        hit_rate_threshold = min(
            hit_rate_threshold,
            float(eligible["direction_accuracy"].median()) - hit_rate_relative_margin,
        )
        logger.info(
            "ドリフトチェック: 実効HitRate閾値=%.1f%% (中央値-%s と上限の小さい方)",
            hit_rate_threshold * 100,
            hit_rate_relative_margin,
        )
    triggered = eligible[
        (eligible["mean_abs_error"] >= mae_threshold)
        | (eligible["direction_accuracy"] <= hit_rate_threshold)
    ]
    record_drift_warn_streak(not triggered.empty)  # NF-303-3 の連続回数（#934）

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
