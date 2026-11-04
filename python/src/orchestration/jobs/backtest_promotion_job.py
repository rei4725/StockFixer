"""バックテストによる仮の入れ替えの夜間ジョブ。

予想のコードの版が前回の評価から変わり、stock_features もその版で作り直された夜に、
取っておいた直近の期間を外して候補を学習し、その期間で今の production と比べる。勝てば候補を本番の名前で置き、5営業日見張る。
見張りで悪ければ、戻す先のほうが良いときだけ写しから戻す。

全体を統合モデルのロックの中で動かし、週次の学習と同時に動かない。
結果は results/backtest_promotion/YYYYMMDD_HHMMSS.json に書き、Discord へ知らせる。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime
from typing import Any, Optional

import pandas as pd
from filelock import Timeout

from src.orchestration.job_errors import NonRetryableJobError
from src.orchestration.jobs.common import _local_now
from src.prediction.backtest_promotion import (
    CANDIDATE_NAMES,
    CODE_VERSION_SCOPE,
    SWAP_PRODUCTION_NAMES,
    WATCH_TRADING_DAYS,
    evaluate_holdout,
    judge_rollback,
    latest_holdout_dates,
    load_features_version,
    load_state,
    prediction_code_version,
    restore_backups,
    results_dir,
    save_state,
    swap_in_candidate,
    unified_model_lock,
)
from src.prediction.baseline_metrics import baseline_metrics
from src.prediction.db import load_prediction_accuracy
from src.prediction.predict_unified import predict_with_unified_model
from src.prediction.types import BacktestPromotionState, HoldoutEvaluation
from src.prediction.unified_model_pipeline import load_unified_model, train_unified_model
from src.utils.japan_time import JAPAN_TIMEZONE, now_jst
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MODEL_TYPES = ["XGBoostModel", "LightGBMModel"]
_LOCK_TIMEOUT_S = 3 * 3600

WON = "won"
LOST = "lost"
NOT_COMPARABLE = "not_comparable"
FAILED = "failed"
SKIPPED = "skipped"


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
    # stock_features は日次の道（平日 07:30）でしか作り直さない。昼に入れた版を古いコードの
    # 特徴量で比べて「比べた」にしないよう、印が今の版になるまで待つ（版は記録しない）
    result["features_version"] = load_features_version()
    if not force and result["features_version"] != version:
        _end(result, SKIPPED, "特徴量がまだ今のコードで作り直されていない")
        return

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
        # 本番の予想が1つも出ず（yfinance の止まりなど）道を確かめられなかった。食い違いではない
        _end(result, FAILED, "道の一致の検査で比べられた銘柄が無い（本番の予想が出ない）")
        return
    if not evaluation.path_check.get("passed"):
        outcome = NOT_COMPARABLE
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
# 見張り
# ---------------------------------------------------------------------------


def _predicted_at_jst(predicted_at: Any) -> Optional[datetime]:
    """predicted_at（プロセスの時計の YYYYMMDD_HHMMSS）を JST の時刻にする。読めなければ None。"""
    try:
        naive = datetime.strptime(str(predicted_at)[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    return naive.replace(tzinfo=_local_now().tzinfo).astimezone(JAPAN_TIMEZONE)


def _scored_rows_since_swap(watch: dict) -> pd.DataFrame:
    """入れ替えの後に予想され、採点が済んだ production（horizon=1）の行。_day に JST の日付。

    predicted_at はプロセスの時計（本番は UTC）の時刻なので、日付でなく時刻で入れ替えの後かを見る。
    22:00 JST に入れ替えた翌朝 07:30 JST の回は、UTC では入れ替えと同じ日付（前日 22:30）になり、
    「predicted_at の日付 > swap_date」で比べるとその回を落とすためである。日数は JST の日付で数える。
    """
    swapped_at = datetime.fromisoformat(watch["swapped_at"])
    since_days = (now_jst().date() - date.fromisoformat(watch["swap_date"])).days + 2
    rows = load_prediction_accuracy(
        horizon=1, limit=0, model_name="production", days=max(since_days, 1), raise_on_error=True
    )
    if rows is None or rows.empty or "actual_ratio" not in rows.columns:
        return pd.DataFrame()
    rows = rows[pd.to_numeric(rows["actual_ratio"], errors="coerce").notna()]
    times = rows["predicted_at"].map(_predicted_at_jst)
    keep = times.map(lambda t: t is not None and t > swapped_at).astype(bool)
    rows = rows[keep]
    return rows.assign(_day=times[keep].map(lambda t: t.strftime("%Y-%m-%d")))


def _check_watch(state: BacktestPromotionState, result: dict) -> None:
    """採点済みの日が5日たまっていれば1回だけ判定し、見張りを消す。たまっていなければ何もしない。"""
    watch = state.watch
    assert watch is not None
    rows = _scored_rows_since_swap(watch)
    days = sorted(rows["_day"].unique()) if not rows.empty else []
    verdict: dict = {"version": watch.get("version"), "scored_days": len(days), "judged": False}
    result["watch"] = verdict
    if len(days) < WATCH_TRADING_DAYS:
        logger.info(
            "仮の入れ替えの見張り: 採点済み %d 日（%d 日で判定）", len(days), WATCH_TRADING_DAYS
        )
        return

    # 入れ替えの後の、最初の5日だけで測る
    first_days = set(days[:WATCH_TRADING_DAYS])
    live = baseline_metrics(rows[rows["_day"].isin(first_days)])
    decision = judge_rollback(live, watch["backtest_candidate"], watch["backtest_old_production"])
    verdict.update(
        judged=True,
        days=sorted(first_days),
        live=live,
        triggered=decision.triggered,
        old_is_better=decision.old_is_better,
        rolled_back=False,
        backups=watch.get("backups", []),
    )
    if decision.rollback:
        restore_backups(watch["backups"])
        verdict["rolled_back"] = True
        if watch["version"] not in state.rolled_back_versions:
            state.rolled_back_versions.append(watch["version"])
    elif decision.triggered:
        verdict["note"] = "見張りに当たったが戻す先も悪いため戻さない"
    state.watch = None
    save_state(state)
    logger.info(
        "仮の入れ替えの見張りを終えた: rolled_back=%s triggered=%s",
        verdict["rolled_back"],
        decision.triggered,
    )


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
