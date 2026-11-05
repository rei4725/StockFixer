"""週次モデル学習ジョブ（Shadow 評価 → 昇格ゲート → Challenger 再学習）。

weekly.py から分けた（#1295）。weekly.py から re-export している。
"""

from src.orchestration.job_errors import NonRetryableJobError
from src.orchestration.jobs.common import _handle_stage_error, _notify_stage_failure
from src.orchestration.types import PipelineStage
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SHADOW_METRIC_KEYS = (
    "production_hit_rate",
    "production_sharpe",
    "challenger_hit_rate",
    "challenger_sharpe",
)


def _challengers_trained_this_week(challenger_names: list[str]) -> list[str]:
    """今週（UTC の ISO 週）に学習し直された challenger の名前を返す（#905）。

    今週の学習で作られた challenger は、影の予想を経ていない。
    同じ週のやり直しで、影の比べは先週の記録を読むため、それを昇格させないために使う。
    ファイルが無い challenger は含めない。
    """
    import os
    from datetime import datetime, timedelta, timezone

    from src.utils.data_path_utils import get_models_dir

    now = datetime.now(timezone.utc)
    week_start = (now - timedelta(days=now.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    unified_dir = os.path.join(get_models_dir(), "unified")
    fresh = []
    for name in challenger_names:
        path = os.path.join(unified_dir, f"{name}.joblib")
        if not os.path.exists(path):
            continue
        mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
        if mtime >= week_start:
            fresh.append(name)
    return fresh


def _save_shadow_record(record: dict) -> None:
    """週の影の比べの記録を書く。失敗は知らせて週の学習を続ける（#1503）。"""
    try:
        from src.prediction import shadow_record

        shadow_record.save_weekly_shadow_record(record)
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, "影の比べの記録の保存", e)
        _notify_stage_failure("週次学習 影の比べの記録の保存", e)


def _shadow_metrics_missing(result: dict) -> bool:
    """影の比べの指標に欠けがあり、勝ち負けを決められなかった週なら True を返す（#1589）。"""
    return any(result[key] is None for key in _SHADOW_METRIC_KEYS)


def _record_no_comparison(record: dict, result: dict) -> None:
    """比べられなかった週を no_shadow_data と記録し、ファイルが在れば1回知らせる（#1589）。

    challenger のファイルが無い週は比べるものが無いだけなので、知らせない。
    """
    detail = (
        f"件数=production {result.get('n_production')}, challenger {result.get('n_challenger')} "
        f"日数={result.get('window_days')} challenger_trained_at="
        f"{result.get('challenger_trained_at')}"
    )
    logger.warning("[2/4] 影の比べの指標が欠けて比べられなかった: %s", detail)
    record["gate_skipped_reason"] = "no_shadow_data"
    if result.get("challenger_trained_at") is not None:
        _notify_stage_failure(
            "週次学習 [1/4] 影の比べ（比べられなかった）",
            RuntimeError(f"challenger のファイルは在るが指標が欠けた。再学習は続ける。{detail}"),
        )


def run_weekly_training():
    """
    週次実行: Shadow 評価 → 昇格ゲート → Challenger 再学習 → 精度チェック → Discord通知

    全体を統合モデルのロックの中で動かし、バックテストによる仮の入れ替えの夜間ジョブと
    同時に動かない。ロックが取れなければ Discord へ知らせ、ロックなしでは学習しない。
    そのときはジョブを失敗にして同じ日の補完に乗せる。同じ回の中では待ち直さない（#1564）。
    """
    from filelock import Timeout

    from src.prediction.backtest_promotion.state import unified_model_lock

    lock = unified_model_lock()
    try:
        lock.acquire()
    except Timeout as e:
        logger.error("週次学習: 統合モデルのロックが取れないため学習しない: %s", e)
        _notify_stage_failure("週次学習 統合モデルのロック（学習しなかった）", e)
        raise NonRetryableJobError(f"週次学習: 統合モデルのロックが取れない: {e}") from e
    try:
        _run_weekly_training_locked()
    finally:
        lock.release()


def _clear_backtest_watch() -> None:
    """週次の昇格で production が変わったら、仮の入れ替えの見張りをやめる（戻す先が無くなる）。"""
    try:
        from src.prediction.backtest_promotion.state import clear_watch

        clear_watch("週次の昇格")
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, "仮の入れ替えの見張りの解除", e)
        _notify_stage_failure("週次学習 仮の入れ替えの見張りの解除", e)


def _run_weekly_training_locked():
    """
    run_weekly_training の本体（統合モデルのロックを持って呼ぶ）。

    流れ:
        1. Shadow 評価: 前週の production vs challenger の Hit Rate / Sharpe を比較
        2. 昇格ゲート: 全基準（Net Return/MDD/Sharpe/Hit Rate/Slippage）を判定
        3. 昇格: challenger_wins かつ eligible なら Challenger → production に昇格
        4. Challenger 再学習（新しい challenger を学習して次週の評価に備える）
        5. 予測精度チェック & ドリフト警告
        6. Discord 通知（昇格結果含む）

    1・2 の失敗は Discord へ1回知らせて 4 へ進む（勝った challenger が知られずに
    再学習で上書きされないようにする。#1295）。指標が欠けて比べられなかった週は負けと記録せず、
    challenger のファイルが在れば1回知らせて 4 へ進む（#1589）。
    """
    logger.info("=== 週次モデル学習開始 ===")

    from config.settings import AUTO_PROMOTE_MODEL
    from src.prediction.promotion_gate import evaluate_promotion, save_promotion_result
    from src.prediction.shadow_evaluation import (
        _UNIFIED_CHALLENGER_NAMES,
        _UNIFIED_PRODUCTION_NAMES,
        DEFAULT_EVALUATION_DAYS,
        evaluate_shadow_models,
        promote_unified_challenger,
    )
    from src.prediction.shadow_record import apply_shadow_result, new_weekly_shadow_record
    from src.prediction.unified_model_pipeline import train_unified_model

    # 影の比べと昇格の結果を毎週1本の JSON に残す（#1503）
    record = new_weekly_shadow_record(days=DEFAULT_EVALUATION_DAYS)

    # 1. Shadow 評価（NON_CRITICAL: 失敗は知らせて継続）
    logger.info("[1/4] Shadow 評価開始")
    shadow_results = []
    shadow_failed = False
    for prod_name, chal_name in zip(_UNIFIED_PRODUCTION_NAMES, _UNIFIED_CHALLENGER_NAMES):
        try:
            # DB が読めない週を「challenger は勝たなかった」にしない（#1295）
            result = evaluate_shadow_models(
                production_version="production",
                challenger_version="challenger",
                raise_on_error=True,
            )
            shadow_results.append((prod_name, chal_name, result))
            apply_shadow_result(record, result)
            logger.info(
                "Shadow 評価: production=hit=%s sharpe=%s / challenger=hit=%s sharpe=%s / wins=%s",
                result["production_hit_rate"],
                result["production_sharpe"],
                result["challenger_hit_rate"],
                result["challenger_sharpe"],
                result["challenger_wins"],
            )
        except Exception as e:
            _handle_stage_error(PipelineStage.NON_CRITICAL, "Shadow 評価", e)
            _notify_stage_failure("週次学習 [1/4] Shadow 評価", e)
            shadow_failed = True
            record["shadow_failed"] = True
            record["shadow_error"] = str(e)[:1000]
        break  # prediction_accuracy の model_name は "production"/"challenger" で共通なので1回でよい

    # 2 & 3. 昇格ゲート + 昇格（RECOVERABLE: 昇格不可でも後続の再学習は継続）
    promoted = False
    gate_result = None
    fresh_challengers = _challengers_trained_this_week(list(_UNIFIED_CHALLENGER_NAMES))
    record["fresh_challengers"] = fresh_challengers
    if shadow_results and shadow_results[0][2]["challenger_wins"] and fresh_challengers:
        logger.warning(
            "[2/4] 昇格を飛ばす: challenger が今週学習し直されており影の比べを経ていない"
            "（同じ週のやり直しの見込み） fresh=%s",
            fresh_challengers,
        )
        record["gate_skipped_reason"] = "challenger_trained_this_week"
    elif shadow_results and shadow_results[0][2]["challenger_wins"]:
        logger.info("[2/4] 昇格ゲート評価開始")
        record["gate_evaluated"] = True
        try:
            gate_result = evaluate_promotion(
                shadow_model_name="challenger",
                current_model_name="production",
                require_manual_approval=not AUTO_PROMOTE_MODEL,
            )
            record["eligible"] = gate_result.eligible
            record["reason"] = gate_result.reason
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
                    record["promoted"] = True
                    _clear_backtest_watch()
                else:
                    logger.warning("昇格対象ファイルなし: skipped=%s", promote_result["skipped"])
            elif gate_result.eligible:
                logger.info("[3/4] 昇格基準クリアだが AUTO_PROMOTE_MODEL=false のため手動承認待ち")
            else:
                logger.info("昇格ゲート未達（再学習のみ実施）: %s", gate_result.reason)
        except Exception as e:
            _handle_stage_error(PipelineStage.RECOVERABLE, "昇格ゲート評価", e)
            _notify_stage_failure("週次学習 [2/4] 昇格ゲート評価", e)
            record["gate_failed"] = True
    elif shadow_results and _shadow_metrics_missing(shadow_results[0][2]):
        _record_no_comparison(record, shadow_results[0][2])
    elif shadow_results:
        logger.info("[2/4] challenger_wins=False のため昇格ゲートをスキップ")
        record["gate_skipped_reason"] = "challenger_lost"
    elif shadow_failed:
        logger.warning("[2/4] Shadow 評価が失敗したため昇格ゲートをスキップ")
        record["gate_skipped_reason"] = "shadow_failed"
    else:
        logger.info("[2/4] Shadow 評価データなし（初回実行）のため昇格ゲートをスキップ")
        record["gate_skipped_reason"] = "no_shadow_data"
    _save_shadow_record(record)

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
