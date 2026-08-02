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


def fetch_previous_run_stats(model_version: str = "production") -> Optional["PredictionRunStats"]:
    """保存前の前回ラン統計を取得する。

    prediction_results は Delete-Insert 方式のスナップショットテーブルのため、
    今回の予測結果を保存する **前** に呼び出すこと（呼び出し順序の保証は
    daily.py 側の責務）。保存後に呼ぶと今回の保存で前回ランの行が上書き
    済みとなり、前回ランが見つからなくなる（C-1 対策）。

    取得に失敗しても致命的ではないため、例外を送出せず None を返す
    （急変チェックがスキップされるだけ）。
    """
    try:
        from src.prediction.db import load_latest_prediction_timestamp
        from src.prediction.db.prediction_results import load_run_stats_at
        from src.prediction.output_invariants import build_run_stats

        previous_at = load_latest_prediction_timestamp(model_version=model_version)
        if not previous_at:
            return None
        previous_raw = load_run_stats_at(previous_at, model_version=model_version)
        if previous_raw is None:
            return None
        return build_run_stats(previous_raw[0], previous_raw[1])
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
    """
    from src.prediction.output_invariants import evaluate_output_invariants

    report = evaluate_output_invariants(
        requested_model_names=requested_models,
        loaded_model_names=loaded_models,
        output_rows=output_rows,
        previous_stats=previous_stats,
    )
    return report.violation_ids, report.as_details()


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
