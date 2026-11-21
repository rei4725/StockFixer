"""バックテストによる仮の入れ替えの夜間ジョブ。

予想のコードの版が前回の評価から変わり、stock_features もその版で作り直された夜に、
取っておいた直近の期間を外して候補を学習し、その期間で今の production と比べる。勝てば候補を本番の名前で置き、5営業日見張る。
見張りで悪ければ、同じ5日の行で旧 production（写し）のほうが良いときだけ写しから戻す。
見張りは backtest_promotion_watch にある。

全体を統合モデルのロックの中で動かし、週次の学習と同時に動かない。
結果は results/backtest_promotion/YYYYMMDD_HHMMSS.json に書き、Discord へ知らせる。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Optional

from filelock import Timeout

from src.orchestration.job_errors import NonRetryableJobError
from src.orchestration.jobs.backtest_promotion_watch import _check_watch
from src.orchestration.jobs.features_wait import WAIT_NEXT_DAILY_RUN, classify_features_wait
from src.prediction.backtest_promotion import (
    CANDIDATE_NAMES,
    CODE_VERSION_SCOPE,
    FEATURES_CODE_SCOPE,
    SWAP_PRODUCTION_NAMES,
    evaluate_holdout,
    features_code_version,
    latest_holdout_dates,
    load_features_version,
    load_features_written_at,
    load_state,
    prediction_code_version,
    restore_backups,
    results_dir,
    save_state,
    swap_in_candidate,
    unified_model_lock,
)
from src.prediction.predict_unified import predict_with_unified_model
from src.prediction.types import BacktestPromotionState, HoldoutEvaluation
from src.prediction.unified_model_pipeline import load_unified_model, train_unified_model
from src.utils.japan_time import now_jst
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MODEL_TYPES = ["XGBoostModel", "LightGBMModel"]
_LOCK_TIMEOUT_S = 3 * 3600

WON = "won"
LOST = "lost"
NOT_COMPARABLE = "not_comparable"
FAILED = "failed"
SKIPPED = "skipped"

STALE_FEATURES_REASON = "特徴量がまだ今のコードで作り直されていない"
# 特徴量の待ちで続けて飛んだ夜がこの数に届いたら WARNING と知らせを出す（#2014）
STALE_FEATURES_ALERT_NIGHTS = 2


def run_backtest_promotion(force: bool = False, dry_run: bool = False) -> dict:
    """見張りの判定 → 評価 → 勝てば入れ替え、を1回動かす。

    Args:
        force: 同じ版・戻した版・特徴量がまだ古い版でも評価し直す
            （戻した版からは入れ替えない。見張り中は評価しない）
        dry_run: 評価だけして、モデルのファイルも状態も変えない（見張りの判定もしない）

    Returns:
        結果の dict。outcome は won・lost・not_comparable・failed・skipped。
        JSON を書いたときは result_path にそのパス（書かなかったときは None）
    """
    started = now_jst()
    result = _new_result(started, force, dry_run)
    lock = unified_model_lock(_LOCK_TIMEOUT_S)
    try:
        lock.acquire()
    except Timeout as e:
        logger.warning("統合モデルのロックが取れないため見送る（週次の学習が動いている見込み）")
        result["lock_timeout"] = True
        _end(result, SKIPPED, f"統合モデルのロックが取れない（{_LOCK_TIMEOUT_S} 秒待った）: {e}")
    else:
        try:
            _run_locked(result, force, dry_run)
        except Exception as e:
            # 状態のファイルが読めない・版が出せないなども、黙って落とさず失敗として知らせる
            logger.error("仮の入れ替えの夜間ジョブに失敗", exc_info=True)
            _end(result, FAILED, "状態の読み書きか版の計算に失敗", e)
        finally:
            lock.release()

    if _should_record(result):
        result["result_path"] = _write_result(result, started)
    _notify(result)
    logger.info(
        "仮の入れ替えの夜間ジョブ: outcome=%s reason=%s path=%s",
        result["outcome"],
        result["reason"],
        result["result_path"],
    )
    return result


def raise_if_backtest_promotion_failed(result: dict) -> None:
    """失敗とロック切れの回をスケジューラの失敗にする（#1564）。手の CLI は呼ばない。

    run_backtest_promotion は JSON と知らせのために結果を dict で返すので、
    定時の回は戻り値を見て投げる。やり直すと学習が頭から走るのでリトライはさせない。
    """
    if result["outcome"] != FAILED and not result["lock_timeout"]:
        return
    detail = f"{result['reason']}（{result['error']}）" if result.get("error") else result["reason"]
    raise NonRetryableJobError(f"仮の入れ替えの夜間ジョブが {result['outcome']}: {detail}")


def _new_result(started: datetime, force: bool, dry_run: bool) -> dict:
    return {
        "started_at": started.isoformat(),
        "version": None,
        "version_scope": list(CODE_VERSION_SCOPE),
        "features_version": None,
        "features_code_version": None,
        "features_scope": list(FEATURES_CODE_SCOPE),
        "stale_features_nights": None,
        "stale_features_cause": None,
        "next_daily_run": None,
        "force": force,
        "dry_run": dry_run,
        "outcome": SKIPPED,
        "reason": None,
        "error": None,
        "lock_timeout": False,
        "watch": None,
        "cutoff": None,
        "dates": None,
        "evaluation": None,
        "swapped": False,
        "backups": [],
        "swap_blocked_reason": None,
        "swap_undone": None,
        "restore_error": None,
        "result_path": None,
    }


def _end(result: dict, outcome: str, reason: Optional[str], error: Optional[Exception] = None):
    result["outcome"] = outcome
    result["reason"] = reason
    if error is not None:
        result["error"] = f"{type(error).__name__}: {error}"


def _run_locked(result: dict, force: bool, dry_run: bool) -> None:
    state = load_state()

    # 1. 見張り（dry-run では判定しない）
    if state.watch is not None and not dry_run:
        try:
            _check_watch(state, result)
        except Exception as e:
            logger.error("仮の入れ替えの見張りの判定に失敗", exc_info=True)
            # 判定の途中までの結果（戻したかどうか）は消さずに残す
            verdict = result["watch"] if result["watch"] is not None else {"judged": False}
            verdict["error"] = f"{type(e).__name__}: {e}"
            result["watch"] = verdict
            if verdict.get("rolled_back"):
                reason = (
                    "見張りで production を旧モデルに戻したが、状態を保存できなかった"
                    "（state.json には見張りが残る）"
                )
            else:
                reason = "見張りの判定に失敗（見張りは残す）"
            _end(result, FAILED, reason, e)
            return
    if state.watch is not None:
        _end(result, SKIPPED, "見張り中のため評価しない")
        return

    # 2. 評価するか
    version = prediction_code_version()
    result["version"] = version
    if not force and version == state.last_evaluated_version:
        _end(result, SKIPPED, "この版はもう比べた")
        return
    if not force and version in state.rolled_back_versions:
        _end(result, SKIPPED, "この版の候補は前に戻した")
        return
    # stock_features は日次の道（平日 07:30）でしか作り直さない。昼に入れた特徴量のコードを
    # 古いコードの特徴量で比べて「比べた」にしないよう、印が今の特徴量の版になるまで待つ
    # （版は記録しない）。特徴量に関わらない直しでは待たない（#2014）
    result["features_version"] = load_features_version()
    result["features_code_version"] = features_code_version()
    if not force and result["features_version"] != result["features_code_version"]:
        _end(result, SKIPPED, STALE_FEATURES_REASON)
        wait = classify_features_wait(load_features_written_at(), now_jst())
        result["stale_features_cause"] = wait.cause
        result["next_daily_run"] = wait.next_daily_run.isoformat()
        if wait.cause == WAIT_NEXT_DAILY_RUN:
            _log_waiting_next_daily_run(state, result)
        else:
            _count_stale_features(state, result)
        return
    if state.stale_features_skips:
        state.stale_features_skips = 0
        save_state(state)

    # 3〜6. 学習・評価・入れ替え・版の記録。ここで落ちたら版は記録しない（次の夜にやり直す）
    try:
        evaluation = _train_and_evaluate(result)
        _apply_evaluation(result, evaluation)
        if dry_run or result["outcome"] == FAILED:
            # 比べられなかった回は版を記録しない（次の夜にやり直す）
            return
        if result["outcome"] == WON:
            if version in state.rolled_back_versions:
                result["swap_blocked_reason"] = "この版の候補は前に戻したため入れ替えない"
            else:
                _swap(state, result, evaluation, version)
        state.last_evaluated_version = version
        save_state(state)
    except Exception as e:
        logger.error("仮の入れ替えの評価に失敗", exc_info=True)
        _end(result, FAILED, "学習か評価か入れ替えに失敗（負けではない）", e)
        if result["swapped"]:
            # 入れ替えた後に見張りを保存できなかった。見張りの無い候補を本番に残さない
            _undo_swap(result)


def _log_waiting_next_daily_run(state: BacktestPromotionState, result: dict) -> None:
    """印の後に特徴量のコードが変わり、次の日次の道を待つだけの夜。INFO だけ残し晩に数えない。

    印を書いた後に日次の道の時刻が来ていないので、前に数えた晩は続いていない（#2086）。
    """
    if state.stale_features_skips:
        state.stale_features_skips = 0
        save_state(state)
    result["stale_features_nights"] = 0
    logger.info(
        "仮の入れ替えは特徴量の待ち: 印の後に特徴量のコードが変わり、次の日次の道 %s を待つ"
        "（特徴量の印 %s・今の特徴量の版 %s）",
        result["next_daily_run"],
        result["features_version"],
        result["features_code_version"],
    )


def _count_stale_features(state: BacktestPromotionState, result: dict) -> None:
    """日次の道が走ったのに印を書けなかった夜を数え、線の晩に届いたら WARNING と知らせ。

    数えた夜は state.json に残す。知らせを送れなくてもジョブは落とさない（#2014）。
    """
    from src.reporting.discord import notifications_backtest_watch as nw

    state.stale_features_skips += 1
    result["stale_features_nights"] = state.stale_features_skips
    save_state(state)
    if state.stale_features_skips < STALE_FEATURES_ALERT_NIGHTS:
        return
    logger.warning(
        "仮の入れ替えが特徴量の待ちで %d 晩続けて飛んだ。日次の道が走ったのに印を書けていない"
        "（特徴量の印 %s・今の特徴量の版 %s・予想の版 %s）",
        state.stale_features_skips,
        result["features_version"],
        result["features_code_version"],
        result["version"],
    )
    try:
        nw.send_backtest_promotion_stale_features(result)
    except Exception:
        logger.error("仮の入れ替えの、特徴量の待ちの通知に失敗", exc_info=True)


def _undo_swap(result: dict) -> None:
    """入れ替えた後に状態を保存できなかったとき、写しから本番を戻す。"""
    backups = result["backups"]
    try:
        restore_backups(backups)
    except Exception as e:
        logger.error(
            "状態の保存に失敗し、入れ替えも戻せなかった。production が候補のままの恐れ: 写し %s",
            backups,
            exc_info=True,
        )
        result["swap_undone"] = False
        result["restore_error"] = f"{type(e).__name__}: {e}"
        result["reason"] = (
            "入れ替えた後に状態を保存できず、戻しにも失敗した。production が候補のままの恐れ"
        )
        result["error"] = f"{result['error']}\n戻しの失敗: {result['restore_error']}"
        return
    logger.warning("入れ替えた後に状態を保存できなかったため、写しから本番を戻した: %s", backups)
    result["swapped"] = False
    result["swap_undone"] = True
    result["reason"] = "入れ替えた後に状態を保存できなかったため、入れ替えを戻した（負けではない）"


def _train_and_evaluate(result: dict) -> HoldoutEvaluation:
    dates = latest_holdout_dates()
    cutoff = dates[0]
    result["cutoff"] = cutoff.strftime("%Y-%m-%d")
    result["dates"] = [d.strftime("%Y-%m-%d") for d in dates]

    # 候補は締め日より前だけで学び、保存済みの除外特徴量を使わず、選択の結果も保存しない
    for model_type, name in zip(_MODEL_TYPES, CANDIDATE_NAMES):
        logger.info("候補の学習: %s（締め日 %s）", name, result["cutoff"])
        train_unified_model(
            model_type=model_type,
            model_name=name,
            train_end_date=cutoff,
            apply_saved_exclusions=False,
            save_selection=False,
        )

    # 候補は本番の名前で重みを引く
    candidate_models = [
        (prod, load_unified_model(cand))
        for prod, cand in zip(SWAP_PRODUCTION_NAMES, CANDIDATE_NAMES)
    ]
    production_models = [(prod, load_unified_model(prod)) for prod in SWAP_PRODUCTION_NAMES]
    return evaluate_holdout(
        cutoff_dates=dates,
        candidate_models=candidate_models,
        production_models=production_models,
        live_predict=predict_with_unified_model,
    )


def _apply_evaluation(result: dict, evaluation: HoldoutEvaluation) -> None:
    result["evaluation"] = asdict(evaluation)
    if not evaluation.path_check.get("compared"):
        # 本番の予想が1つも出ず道を確かめられなかった。食い違いではない
        _end(result, FAILED, "道の一致の検査で比べられた銘柄が無い（本番の予想が出ない）")
        return
    if not (evaluation.path_check.get("compared_by_market") or {}).get("jp"):
        # 発注する市場を確かめられなかった回は入れ替えず、版を記録せず次の夜にやり直す（#1675）
        _end(result, FAILED, "道の一致の検査で jp の比べられた銘柄が無い（jp の予想が出ない）")
        return
    if not evaluation.path_check.get("passed"):
        outcome = NOT_COMPARABLE
    elif evaluation.required_market_shortfalls:
        # 発注する市場の行か銘柄か日が線の下の回は負けと記録せず、版を残さず次の夜に比べ直す（#1698）
        outcome = FAILED
    elif evaluation.decision.won:
        outcome = WON
    else:
        outcome = LOST
    _end(result, outcome, "; ".join(evaluation.decision.reasons) or None)


def _swap(
    state: BacktestPromotionState, result: dict, evaluation: HoldoutEvaluation, version: str
) -> None:
    backups = swap_in_candidate()
    result["swapped"] = True
    result["backups"] = backups
    swapped_at = now_jst()
    state.watch = {
        "version": version,
        "swapped_at": swapped_at.isoformat(),
        "swap_date": swapped_at.strftime("%Y-%m-%d"),
        "backups": backups,
        "backtest_candidate": evaluation.candidate,
        "backtest_old_production": evaluation.production,
    }


# ---------------------------------------------------------------------------
# 記録と通知
# ---------------------------------------------------------------------------


def _should_record(result: dict) -> bool:
    """評価した・失敗した・見張りを判定した・ロックで見送った回だけ JSON と通知を出す。"""
    watch = result.get("watch") or {}
    return result["outcome"] != SKIPPED or result["lock_timeout"] or bool(watch.get("judged"))


def _write_result(result: dict, started: datetime) -> str:
    folder = results_dir()
    folder.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%d_%H%M%S")
    path = folder / f"{stamp}.json"
    n = 2
    while path.exists():
        path = folder / f"{stamp}_{n}.json"
        n += 1
    result["result_path"] = str(path)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("仮の入れ替えの結果を保存: %s", path)
    return str(path)


def _notify(result: dict) -> None:
    from src.reporting.discord import notifications_model as nm

    watch = result.get("watch") or {}
    sends = []
    if watch.get("judged"):
        sends.append(("見張り", nm.send_backtest_promotion_rollback))
    if result["outcome"] != SKIPPED or result["lock_timeout"]:
        sends.append(("評価の結果", nm.send_backtest_promotion_result))
    if result["swapped"] and result["outcome"] != FAILED:
        sends.append(("入れ替え", nm.send_backtest_promotion_swap))
    for label, send in sends:
        try:
            send(result)
        except Exception:
            logger.error("仮の入れ替えの%sの通知に失敗", label, exc_info=True)


__all__ = ["raise_if_backtest_promotion_failed", "run_backtest_promotion"]
