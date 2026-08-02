"""条件付きアラートルール定義（NF-303）

アラート条件:
  - 日次パイプライン 2 回以上連続失敗
  - 日次損失上限（R-003）が 3 日連続発動
  - ドリフト検知（R-104）が Warn 以上を連続検出
  - /health が degraded 継続（NF-301 前提）
  - 予測出力の健全性（出力 invariant）違反

条件成立時: Discord に詳細アラートを送信
条件非成立時: 何も送信しない
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field

from src.utils.logger import get_logger

logger = get_logger(__name__)

NotifierFn = Callable[[str, str, int], bool]

# ---------------------------------------------------------------------------
# 閾値定数
# ---------------------------------------------------------------------------

PIPELINE_FAIL_THRESHOLD = 2  # 連続失敗回数
LOSS_LIMIT_THRESHOLD = 3  # 連続損失上限発動日数
DRIFT_WARN_THRESHOLD = 2  # 連続ドリフト警告検出回数（Warn 以上）
HEALTH_DEGRADED_THRESHOLD = 2  # 連続 degraded 確認回数
PREDICTION_OUTPUT_THRESHOLD = 1  # 予測出力 invariant はストリークを使わず即発報

# ---------------------------------------------------------------------------
# system_config キー（ストリーク永続化）
# ---------------------------------------------------------------------------

_KEY_LOSS_LIMIT_STREAK = "alert.loss_limit_streak"
_KEY_DRIFT_WARN_STREAK = "alert.drift_warn_streak"
_KEY_HEALTH_DEGRADED_STREAK = "alert.health_degraded_streak"

# スケジューラ日次パイプラインのジョブID
_DAILY_PIPELINE_JOB_ID = "daily_pipeline"


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------


@dataclass
class AlertResult:
    """1 つのアラートルール評価結果。"""

    rule_id: str
    name: str
    triggered: bool
    consecutive_count: int
    threshold: int
    details: dict = field(default_factory=dict)

    def as_discord_lines(self) -> list[str]:
        status = "🚨 **アラート**" if self.triggered else "✅ 正常"
        lines = [
            f"{status} [{self.rule_id}] {self.name}",
            f"  連続回数: {self.consecutive_count} / 閾値: {self.threshold}",
        ]
        for violation in self.details.get("violations", []):
            lines.append(
                f"  - [{violation.get('id')}] {violation.get('description')} "
                f"(実測 {violation.get('observed')} / 閾値 {violation.get('threshold')})"
            )
        return lines


# ---------------------------------------------------------------------------
# 内部ヘルパー：system_config ストリーク操作
# ---------------------------------------------------------------------------


def _get_streak(key: str) -> int:
    try:
        from src.utils.db.system_config import get_config_value

        value = get_config_value(key, "0")
        return int(value or "0")
    except Exception as exc:
        logger.error("ストリーク取得失敗 key=%s: %s", key, exc, exc_info=True)
        return 0


def _set_streak(key: str, value: int) -> None:
    try:
        from src.utils.db.system_config import set_config_value

        set_config_value(key, str(value))
    except Exception as exc:
        logger.error("ストリーク更新失敗 key=%s: %s", key, exc, exc_info=True)


# ---------------------------------------------------------------------------
# ルール 1: 日次パイプライン連続失敗
# ---------------------------------------------------------------------------


def _count_consecutive_pipeline_failures(state_file_path: str | None = None) -> int:
    """scheduler_queue_state.json から daily_pipeline の連続失敗回数を返す。"""
    if state_file_path is None:
        from src.utils.data_path_utils import get_results_dir

        state_file_path = os.path.join(get_results_dir(), "scheduler_queue_state.json")

    if not os.path.exists(state_file_path):
        return 0

    try:
        with open(state_file_path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("スケジューラ状態ファイル読込失敗: %s", exc)
        return 0

    events = [
        e
        for e in state.get("events", [])
        if e.get("job_id") == _DAILY_PIPELINE_JOB_ID and e.get("status") in {"success", "error"}
    ]
    events.sort(key=lambda e: e.get("finished_at", ""), reverse=True)

    consecutive = 0
    for event in events:
        if event.get("status") == "error":
            consecutive += 1
        else:
            break

    return consecutive


def check_pipeline_fail_rule(state_file_path: str | None = None) -> AlertResult:
    """日次パイプライン連続失敗ルールを評価する。"""
    count = _count_consecutive_pipeline_failures(state_file_path)
    triggered = count >= PIPELINE_FAIL_THRESHOLD
    logger.info("パイプライン連続失敗チェック: count=%d triggered=%s", count, triggered)
    return AlertResult(
        rule_id="NF-303-1",
        name="日次パイプライン連続失敗",
        triggered=triggered,
        consecutive_count=count,
        threshold=PIPELINE_FAIL_THRESHOLD,
        details={"job_id": _DAILY_PIPELINE_JOB_ID},
    )


# ---------------------------------------------------------------------------
# ルール 2: 日次損失上限連続発動（R-003）
# ---------------------------------------------------------------------------


def update_loss_limit_streak(triggered: bool) -> int:
    """
    損失上限発動の有無に基づきストリークを更新して返す。

    orchestration 層の run_daily_auto_order() から呼び出す。
    """
    current = _get_streak(_KEY_LOSS_LIMIT_STREAK)
    new_value = current + 1 if triggered else 0
    _set_streak(_KEY_LOSS_LIMIT_STREAK, new_value)
    logger.info("損失上限ストリーク更新: %d → %d", current, new_value)
    return new_value


def check_loss_limit_rule() -> AlertResult:
    """日次損失上限連続発動ルールを評価する。"""
    count = _get_streak(_KEY_LOSS_LIMIT_STREAK)
    triggered = count >= LOSS_LIMIT_THRESHOLD
    logger.info("損失上限連続チェック: count=%d triggered=%s", count, triggered)
    return AlertResult(
        rule_id="NF-303-2",
        name="日次損失上限（R-003）連続発動",
        triggered=triggered,
        consecutive_count=count,
        threshold=LOSS_LIMIT_THRESHOLD,
        details={"config_key": _KEY_LOSS_LIMIT_STREAK},
    )


# ---------------------------------------------------------------------------
# ルール 3: ドリフト連続警告（R-104）
# ---------------------------------------------------------------------------


def update_drift_warn_streak(warn_detected: bool) -> int:
    """
    ドリフト Warn 以上検出の有無に基づきストリークを更新して返す。

    orchestration 層の run_daily_drift_check() から呼び出す。
    """
    current = _get_streak(_KEY_DRIFT_WARN_STREAK)
    new_value = current + 1 if warn_detected else 0
    _set_streak(_KEY_DRIFT_WARN_STREAK, new_value)
    logger.info("ドリフト警告ストリーク更新: %d → %d", current, new_value)
    return new_value


def check_drift_warn_rule() -> AlertResult:
    """ドリフト連続警告ルールを評価する。"""
    count = _get_streak(_KEY_DRIFT_WARN_STREAK)
    triggered = count >= DRIFT_WARN_THRESHOLD
    logger.info("ドリフト警告連続チェック: count=%d triggered=%s", count, triggered)
    return AlertResult(
        rule_id="NF-303-3",
        name="ドリフト検知（R-104）連続 Warn",
        triggered=triggered,
        consecutive_count=count,
        threshold=DRIFT_WARN_THRESHOLD,
        details={"config_key": _KEY_DRIFT_WARN_STREAK},
    )


# ---------------------------------------------------------------------------
# ルール 4: /health degraded 継続（NF-301 前提）
# ---------------------------------------------------------------------------


def update_health_degraded_streak(is_degraded: bool) -> int:
    """
    ヘルスチェック degraded の有無に基づきストリークを更新して返す。

    api/health.py のヘルスチェック実行後に呼び出す。
    """
    current = _get_streak(_KEY_HEALTH_DEGRADED_STREAK)
    new_value = current + 1 if is_degraded else 0
    _set_streak(_KEY_HEALTH_DEGRADED_STREAK, new_value)
    logger.info("ヘルス degraded ストリーク更新: %d → %d", current, new_value)
    return new_value


def check_health_degraded_rule() -> AlertResult:
    """/health degraded 継続ルールを評価する。"""
    count = _get_streak(_KEY_HEALTH_DEGRADED_STREAK)
    triggered = count >= HEALTH_DEGRADED_THRESHOLD
    logger.info("ヘルス degraded 継続チェック: count=%d triggered=%s", count, triggered)
    return AlertResult(
        rule_id="NF-303-4",
        name="/health degraded 継続",
        triggered=triggered,
        consecutive_count=count,
        threshold=HEALTH_DEGRADED_THRESHOLD,
        details={"config_key": _KEY_HEALTH_DEGRADED_STREAK},
    )


# ---------------------------------------------------------------------------
# ルール 5: 予測出力の健全性（出力 invariant）
# ---------------------------------------------------------------------------


def check_prediction_output_rule(
    violation_ids: list[str] | None,
    details: dict | None = None,
) -> AlertResult:
    """予測出力の不変条件違反を評価する。

    判定ロジック本体は src/prediction/output_invariants.py にある。本モジュールは
    src.utils 層にあり src.prediction を import できない（.importlinter Contract 1）
    ため、プリミティブのみを受け取る。

    Args:
        violation_ids: 違反 ID のリスト。None は「評価が実行されなかった」を意味し、
            違反として扱う。空リストは「評価した結果、違反なし」を意味する。
        details: 評価の内訳（銘柄数・中央値など）

    Returns:
        AlertResult（rule_id="NF-303-5"）
    """
    payload = dict(details or {})

    if violation_ids is None:
        payload["violation_ids"] = ["A-0"]
        payload["note"] = "出力 invariant の評価が実行されなかった"
        # as_discord_lines() は details["violations"] しか描画しない。ここを
        # 空のままにすると、A-0 発報時に Discord 本文へ理由が一切出ず、運用者は
        # 「連続回数: 1 / 閾値: 1」しか受け取れない（I-3 対策）。
        payload["violations"] = [
            {
                "id": "A-0",
                "description": (
                    "出力 invariant の評価が実行されなかった"
                    "（[2.1/5] 出力 invariant 評価ステージで例外が発生した可能性がある。"
                    "ログを確認すること）"
                ),
                "observed": 0.0,
                "threshold": 1.0,
            }
        ]
        logger.warning("予測出力チェック: 評価が実行されていない（違反として扱う）")
        return AlertResult(
            rule_id="NF-303-5",
            name="予測出力の健全性",
            triggered=True,
            consecutive_count=1,
            threshold=PREDICTION_OUTPUT_THRESHOLD,
            details=payload,
        )

    payload["violation_ids"] = list(violation_ids)
    count = len(violation_ids)
    triggered = count >= PREDICTION_OUTPUT_THRESHOLD
    logger.info("予測出力チェック: violations=%d triggered=%s", count, triggered)
    return AlertResult(
        rule_id="NF-303-5",
        name="予測出力の健全性",
        triggered=triggered,
        consecutive_count=count,
        threshold=PREDICTION_OUTPUT_THRESHOLD,
        details=payload,
    )


# ---------------------------------------------------------------------------
# アラート条件の一括評価
# ---------------------------------------------------------------------------


def evaluate_alert_conditions(
    state_file_path: str | None = None,
    prediction_violation_ids: list[str] | None = None,
    prediction_details: dict | None = None,
) -> list[AlertResult]:
    """
    全アラートルールを評価し、結果リストを返す。

    Args:
        state_file_path: scheduler_queue_state.json のパス（None = デフォルト）
        prediction_violation_ids: 予測出力 invariant の違反 ID。None は未評価を意味する
        prediction_details: 予測出力 invariant の内訳

    Returns:
        AlertResult リスト（triggered=True が優先アラート対象）
    """
    results = [
        check_pipeline_fail_rule(state_file_path),
        check_loss_limit_rule(),
        check_drift_warn_rule(),
        check_health_degraded_rule(),
        check_prediction_output_rule(prediction_violation_ids, prediction_details),
    ]
    triggered_count = sum(1 for r in results if r.triggered)
    logger.info("アラート評価完了: %d/%d 条件成立", triggered_count, len(results))
    return results


# ---------------------------------------------------------------------------
# Discord 通知
# ---------------------------------------------------------------------------


def _send_alert_detail(results: list[AlertResult], notifier: NotifierFn) -> bool:
    """アラート詳細を Discord に送信する。"""
    lines = ["以下の条件が成立しています。確認してください。", ""]
    for r in results:
        if r.triggered:
            lines.extend(r.as_discord_lines())
            lines.append("")

    return notifier("🚨 運用アラート発報（NF-303）", "\n".join(lines).strip(), 0xFF0000)


def run_conditional_notification(
    results: list[AlertResult] | None = None,
    state_file_path: str | None = None,
    notifier: NotifierFn | None = None,
) -> bool:
    """
    アラート条件を評価し、成立時のみ詳細アラートを送信する。

    Args:
        results: 評価済みの AlertResult リスト。None の場合は内部で evaluate する。
        state_file_path: scheduler_queue_state.json のパス（None = デフォルト）
        notifier: Discord 送信コールバック（title, message, color） -> bool。None の場合は通知スキップ。

    Returns:
        Discord 送信成功時 True
    """
    if results is None:
        results = evaluate_alert_conditions(state_file_path)

    if notifier is None:
        return False

    any_triggered = any(r.triggered for r in results)

    if any_triggered:
        logger.warning("アラート条件成立 — 詳細通知を送信します")
        return _send_alert_detail(results, notifier)

    logger.info("アラート条件非成立 — 通知は送信しません")
    return False
