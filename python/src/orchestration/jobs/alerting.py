"""日次パイプラインの出力 invariant 評価・運用アラート発報ヘルパー。

`run_daily_pipeline()`（daily.py）の行数削減のため、以下の非致命的
（NON_CRITICAL）ステージ処理をここへ切り出す:

    - 保存前の前回ラン統計取得（[2/5] の内側）
    - 出力 invariant 評価（[2.1/5]）
    - 運用アラート評価 + Discord 通知（[6/6]）

ステージ分類（PipelineStage）に基づくエラーハンドリングとログ出力は
呼び出し元の daily.py 側に残す。ここでは例外を送出するだけで揉み消さない
（前回ラン統計取得のみ、元の挙動どおり内部で捕捉して None に縮退する）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.prediction.output_invariants import PredictionRunStats
    from src.prediction.types import PredictionResult

logger = get_logger(__name__)

# 同じ日の1回目が読んだ前の日の回の統計（#920）。キーは JST の今日の YYYYMMDD（#935）。
# 日次の道を頭からやり直すと、最新の回は1回目が保存した今日の回になっているため、
# 2回目はここに退避した前の日の回と比べる。
_previous_day_stats: dict[str, "PredictionRunStats"] = {}
# 前の日の回の統計が無く、急変の見張りを飛ばした日（YYYYMMDD）
_compare_skipped_days: set[str] = set()

_COMPARE_SKIPPED_VIOLATION = {
    "id": "B-0",
    "description": (
        "急変の見張り（B-1〜B-3）を飛ばした: production の最新の回が既に今日で、"
        "前の日の回の統計が無い（日次の道のやり直し）"
    ),
    "observed": 0.0,
    "threshold": 1.0,
}


def fetch_previous_run_stats(model_version: str = "production") -> Optional["PredictionRunStats"]:
    """保存前の前回ラン統計を取得する。

    prediction_results は Delete-Insert 方式のスナップショットテーブルのため、
    今回の予測結果を保存する **前** に呼び出すこと（呼び出し順序の保証は
    daily.py 側の責務）。保存後に呼ぶと今回の保存で前回ランの行が上書き
    済みとなり、前回ランが見つからなくなる（C-1 対策）。

    日次の道を頭からやり直して最新の回が既に今日なら、それを前回とせず、
    同じ日の1回目が退避した前の日の回の統計を返す。退避が無ければ None を返し、
    比べを飛ばしたことを evaluate_output_invariants_stage が B-0 として出す（#920）。

    取得に失敗しても致命的ではないため、例外を送出せず None を返す
    （急変チェックがスキップされるだけ）。
    """
    from src.orchestration.jobs.common import is_today_jst, today_jst_key

    today = today_jst_key()
    _compare_skipped_days.discard(today)
    try:
        from src.prediction.db import load_latest_prediction_timestamp
        from src.prediction.db.prediction_results import load_run_stats_at
        from src.prediction.output_invariants import build_run_stats

        previous_at = load_latest_prediction_timestamp(model_version=model_version)
        if not previous_at:
            return None
        # predicted_at は JST に直して JST の今日と比べる（#935。コンテナの時計は UTC）
        if is_today_jst(previous_at):
            cached = _previous_day_stats.get(today)
            if cached is not None:
                logger.warning(
                    "[3/6] production の最新の回が既に今日 (%s) のため、同じ日の1回目が"
                    "退避した前の日の回の統計と比べる（日次の道のやり直し）",
                    previous_at,
                )
                return cached
            _compare_skipped_days.add(today)
            logger.warning(
                "[3/6] production の最新の回が既に今日 (%s) で前の日の回の統計が無いため、"
                "急変の見張り（B-1〜B-3）を飛ばす（日次の道のやり直し）",
                previous_at,
            )
            return None
        previous_raw = load_run_stats_at(previous_at, model_version=model_version)
        if previous_raw is None:
            return None
        stats = build_run_stats(previous_raw[0], previous_raw[1])
        if stats is not None:
            _previous_day_stats.clear()
            _previous_day_stats[today] = stats
        return stats
    except Exception as e:
        logger.error(
            "[2/5] 前回ラン統計の取得に失敗（急変チェックをスキップ）: %s", e, exc_info=True
        )
        return None


def evaluate_output_invariants_stage(
    requested_models: list[str],
    loaded_models: list[str],
    output_rows: list["PredictionResult"],
    previous_stats: Optional["PredictionRunStats"],
) -> tuple[list[str], dict]:
    """出力 invariant を評価し `(violation_ids, details)` を返す。

    例外は呼び出し元（daily.py の [2.1/5] ブロック）へそのまま伝播させる。
    NON_CRITICAL 分類・ログ出力は呼び出し元の責務。
    やり直しで急変の見張りを飛ばした日は、B-0 を足して運用アラートに出す（#920）。
    """
    from src.orchestration.jobs.common import today_jst_key
    from src.prediction.output_invariants import evaluate_output_invariants

    report = evaluate_output_invariants(
        requested_model_names=requested_models,
        loaded_model_names=loaded_models,
        output_rows=output_rows,
        previous_stats=previous_stats,
    )
    violation_ids, details = report.violation_ids, report.as_details()
    if previous_stats is None and today_jst_key() in _compare_skipped_days:
        violation_ids.append(str(_COMPARE_SKIPPED_VIOLATION["id"]))
        details["violations"].append(dict(_COMPARE_SKIPPED_VIOLATION))
    return violation_ids, details


def evaluate_and_notify_alerts(
    prediction_violation_ids: Optional[list[str]], prediction_details: Optional[dict]
) -> None:
    """運用アラート（NF-303）を評価し、条件成立時のみ Discord へ発報する。

    例外は呼び出し元（daily.py の [6/6] ブロック）へそのまま伝播させる。
    NON_CRITICAL 分類・ログ出力は呼び出し元の責務。
    """
    from src.reporting.discord.webhook_sender import send_webhook_notification
    from src.utils.alert_service import evaluate_alert_conditions, run_conditional_notification

    alert_results = evaluate_alert_conditions(
        prediction_violation_ids=prediction_violation_ids,
        prediction_details=prediction_details,
    )
    run_conditional_notification(results=alert_results, notifier=send_webhook_notification)
