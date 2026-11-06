"""
週次ジョブが回数の上限で補完をやめた週の知らせ（#1647, #1666）

should_recover は上限に届くと False を返すだけで、その週の分が抜けたまま古い値で動くことに
気づけない。補完のポーリングが、成功の無いまま上限に届いた期間を見つけたら、何が古いままかを
1期間に1通だけ知らせる。本文はジョブごとに持つ。判定と回数には手を入れない。

送った印は状態ファイルの events に残す。results/ の下にあり、再起動とデプロイをまたいで残る。
印の status は回数・間隔・成功のどの判定にも数えない。
送れなかった回は印を残さず、次のポーリングで送り直す。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, NamedTuple

from src.orchestration.scheduler_cutoff import scheduled_datetime_for_recovery
from src.utils.japan_time import isoformat_utc

if TYPE_CHECKING:
    from src.orchestration.scheduler_queue import SchedulerQueueManager

logger = logging.getLogger("scheduler")

# 週の最適化が、その週に済んだとみなす日数。これより前に最適化した銘柄が残りになる
OPTIMIZATION_SKIP_DAYS = 6

CAP_NOTICE_STATUS = "cap_notice_sent"

# 全銘柄にかかる時間の目安（run_scheduler の SCHEDULE_CONFIG の説明と同じ、#1534）
_FULL_RUN_HOURS = 29


def _capped_period(manager: SchedulerQueueManager, job_id: str, now: datetime) -> str | None:
    """成功の無いまま回数の上限で補完をやめ、まだ知らせていない期間を返す。"""
    config = manager.schedule_config.get(job_id)
    if config is None:
        return None
    with manager._lock:
        if job_id in manager._in_progress_jobs:
            return None

    # 週の定時の前には知らせない。定時の曜日の後も週の終わりまで見る。定時の曜日をまたいで
    # 消えた回の起き直しが翌日でも、その週のうちに知らせるため（#1666）
    scheduled_dt = scheduled_datetime_for_recovery(
        {**config, "recover_rest_of_week": True},
        now,
        manager._parse_day_of_week(config.get("day_of_week", "*")),
    )
    if scheduled_dt is None:
        return None
    if now < scheduled_dt + timedelta(minutes=config.get("recovery_delay_minutes", 10)):
        return None

    if not manager._reload_state_for_recovery():
        return None

    period_key = manager._build_period_key(config, now)
    if (job_id, period_key) in manager._cap_notices_sent or manager.has_success(job_id, period_key):
        return None
    if manager.get_execution_count(job_id, period_key) < config.get("max_executions_per_period", 1):
        return None
    if _period_events(manager, job_id, period_key, CAP_NOTICE_STATUS):
        return None
    return period_key


def _period_events(
    manager: SchedulerQueueManager, job_id: str, period_key: str, status: str
) -> list[dict[str, Any]]:
    return [
        event
        for event in manager._state.get("events", [])
        if event.get("job_id") == job_id
        and event.get("period_key") == period_key
        and event.get("status") == status
    ]


def _load_backlog() -> tuple[int, int, date | None]:
    """残りの銘柄の数・対象の全銘柄の数・残りのうち最も古い最適化の日を返す。"""
    from src.backtest.optimizer.batch import count_optimization_backlog
    from src.watchlist.batch_runner import load_target_symbols

    tasks = load_target_symbols()
    remaining, oldest = count_optimization_backlog(tasks, OPTIMIZATION_SKIP_DAYS)
    return remaining, len(tasks), oldest


def _backlog_line() -> str:
    """残りの見込みの1行を作る。読めなくても知らせは止めない。"""
    try:
        remaining, total, oldest = _load_backlog()
    except Exception:
        logger.error("週の最適化の残りを数えられなかった", exc_info=True)
        return "残りの銘柄は数えられなかった"
    hours = math.ceil(_FULL_RUN_HOURS * remaining / total) if total else 0
    oldest_text = f"、いちばん古い最適化は {oldest.isoformat()}" if oldest else ""
    return f"残り {remaining} / {total} 銘柄（あと約 {hours} 時間の見込み）{oldest_text}"


def _optimization_detail() -> str:
    return (
        f"{_backlog_line()}\n"
        "残りの銘柄は前の週の閾値と SL/TP のまま発注に使われる。"
        "次の週の定時の回が古い順に追いつく"
    )


def _training_detail() -> str:
    return (
        "この週の影の比べ・昇格・challenger の学習・精度のチェックは、"
        "回が止まったところから先が飛ぶ。"
        "昇格まで届かなければ production のモデルは古いまま発注の予想を出す。"
        "補完は定時の曜日だけなので、次の週の定時の回まで動かない"
    )


def _walk_forward_detail() -> str:
    return (
        "Walk-Forward の比べはこの週の分が無い。"
        "補完は定時の曜日だけなので、次の週の定時の回が残っている最後の回と比べる"
    )


class _CapNotice(NamedTuple):
    """知らせの題に入れるジョブの名前と、何が古いままかの本文を作る関数"""

    title: str
    detail: Callable[[], str]


# 知らせるジョブ。補完をやめた後、その週の分が抜けたまま動くジョブである
_CAP_NOTICES = {
    "weekly_optimization": _CapNotice("週の最適化", _optimization_detail),
    "weekly_model_training": _CapNotice("週次モデル学習", _training_detail),
    "weekly_walk_forward_report": _CapNotice("週の Walk-Forward レポート", _walk_forward_detail),
}


def _build_message(manager: SchedulerQueueManager, job_id: str, period_key: str) -> str:
    notice = _CAP_NOTICES[job_id]
    started = len(_period_events(manager, job_id, period_key, "started"))
    failed = len(_period_events(manager, job_id, period_key, "error"))
    return (
        f"⚠️ {notice.title}の補完が回数の上限で止まった（{period_key}）\n"
        f"開始 {started} 回・失敗 {failed} 回で、成功の回が無い\n"
        f"{notice.detail()}"
    )


def _send(message: str) -> bool:
    from src.reporting.discord.discord_utils import send_webhook_text

    return bool(send_webhook_text(message))


def _record_sent(
    manager: SchedulerQueueManager, job_id: str, period_key: str, now: datetime
) -> None:
    # 書けなかったときも、同じ起動の間は5分おきに送り直さない
    manager._cap_notices_sent.add((job_id, period_key))
    stamp = isoformat_utc(now)
    manager._append_event(
        {
            "job_id": job_id,
            "reason": "recovery_cap",
            "status": CAP_NOTICE_STATUS,
            "period_key": period_key,
            "started_at": stamp,
            "finished_at": stamp,
        }
    )


def notify_recovery_cap(manager: SchedulerQueueManager, job_id: str, now: datetime) -> None:
    """上限で補完をやめた週を知らせる。何が落ちても補完のポーリングへ例外を返さない。"""
    if job_id not in _CAP_NOTICES:
        return
    try:
        period_key = _capped_period(manager, job_id, now)
        if period_key is None:
            return
        message = _build_message(manager, job_id, period_key)
        logger.warning(f"回数の上限で補完をやめた週: job_id={job_id} period_key={period_key}")
        if _send(message):
            _record_sent(manager, job_id, period_key, now)
        else:
            logger.error(f"上限の知らせを送れなかった（次のポーリングで送り直す）: {job_id}")
    except Exception:
        logger.error(f"上限の知らせに失敗（次のポーリングへ進む）: {job_id}", exc_info=True)
