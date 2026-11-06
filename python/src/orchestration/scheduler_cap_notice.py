"""
補完をやめた期間の知らせ。週次ジョブの回数の上限（#1647, #1666）と自動発注の締め（#1674）

should_recover は上限や締めで False を返すだけで、その期間の分が抜けたまま動くことに
気づけない。補完のポーリングが、成功の無いまま補完をやめた期間を見つけたら、何が抜けたかを
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

from src.domain.jp_market_holidays import is_jp_business_day
from src.orchestration.scheduler_cutoff import (
    is_past_recovery_until,
    scheduled_datetime_for_recovery,
)
from src.utils.japan_time import isoformat_utc, to_jst

if TYPE_CHECKING:
    from src.orchestration.scheduler_queue import SchedulerQueueManager

logger = logging.getLogger("scheduler")

# 週の最適化が、その週に済んだとみなす日数。これより前に最適化した銘柄が残りになる
OPTIMIZATION_SKIP_DAYS = 6

CAP_NOTICE_STATUS = "cap_notice_sent"

# 全銘柄にかかる時間の目安（run_scheduler の SCHEDULE_CONFIG の説明と同じ、#1534）
_FULL_RUN_HOURS = 29


def _period_without_success(
    manager: SchedulerQueueManager, job_id: str, schedule: dict[str, Any], now: datetime
) -> str | None:
    """補完の待ちを過ぎ、動いておらず、成功も知らせも無い期間を返す。schedule は定時の求め方。"""
    with manager._lock:
        if job_id in manager._in_progress_jobs:
            return None

    scheduled_dt = scheduled_datetime_for_recovery(
        schedule, now, manager._parse_day_of_week(schedule.get("day_of_week", "*"))
    )
    if scheduled_dt is None:
        return None
    if now < scheduled_dt + timedelta(minutes=schedule.get("recovery_delay_minutes", 10)):
        return None

    if not manager._reload_state_for_recovery():
        return None

    period_key = manager._build_period_key(schedule, now)
    if (job_id, period_key) in manager._cap_notices_sent or manager.has_success(job_id, period_key):
        return None
    if _period_events(manager, job_id, period_key, CAP_NOTICE_STATUS):
        return None
    return period_key


def _reached_cap(manager: SchedulerQueueManager, job_id: str, period_key: str) -> bool:
    config = manager.schedule_config[job_id]
    count = manager.get_execution_count(job_id, period_key)
    return bool(count >= config.get("max_executions_per_period", 1))


def _capped_period(manager: SchedulerQueueManager, job_id: str, now: datetime) -> str | None:
    """成功の無いまま回数の上限で補完をやめ、まだ知らせていない期間を返す。"""
    config = manager.schedule_config.get(job_id)
    if config is None:
        return None

    # 週の定時の前には知らせない。定時の曜日の後も週の終わりまで見る。定時の曜日をまたいで
    # 消えた回の起き直しが翌日でも、その週のうちに知らせるため（#1666）
    schedule = {**config, "recover_rest_of_week": True}
    period_key = _period_without_success(manager, job_id, schedule, now)
    if period_key is None or not _reached_cap(manager, job_id, period_key):
        return None
    return period_key


def _missed_day(manager: SchedulerQueueManager, job_id: str, now: datetime) -> str | None:
    """成功の無いまま上限に届いたか締めを過ぎ、まだ知らせていない東証の営業日を返す。"""
    config = manager.schedule_config.get(job_id)
    if config is None:
        return None
    # 休場日の発注の回は何もせず終わるので、回が動かなくても抜けるものが無い
    if not is_jp_business_day(to_jst(now).date()):
        return None

    period_key = _period_without_success(manager, job_id, config, now)
    if period_key is None:
        return None
    # 上限の下で締めの前なら、補完がまだ動ける
    if not _reached_cap(manager, job_id, period_key) and not is_past_recovery_until(config, now):
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


def _auto_order_detail() -> str:
    return (
        "発注の回はやり直さない。次に動くのは次の営業日の 8:50 の定時の回である\n"
        "回が止まった時点より後の SL/TP・売りシグナル・新規買いは出ていないかもしれない"
        "（回が1度も始まっていない日は、発注の回からはどれも出ていない）\n"
        "始まった回があれば、途中までの注文は出ていることがある"
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


# 締めのある日次のジョブ。回数0のまま締めを過ぎた日も知らせる（#1674）
_MISSED_DAY_NOTICES = {
    "daily_auto_order": _CapNotice("自動発注", _auto_order_detail),
}


def _build_message(manager: SchedulerQueueManager, job_id: str, period_key: str) -> str:
    if job_id in _MISSED_DAY_NOTICES:
        notice = _MISSED_DAY_NOTICES[job_id]
        headline = "の回が成功しないまま補完を終えた"
    else:
        notice = _CAP_NOTICES[job_id]
        headline = "の補完が回数の上限で止まった"
    started = len(_period_events(manager, job_id, period_key, "started"))
    failed = len(_period_events(manager, job_id, period_key, "error"))
    return (
        f"⚠️ {notice.title}{headline}（{period_key}）\n"
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
    """補完をやめた期間を知らせる。何が落ちても補完のポーリングへ例外を返さない。"""
    if job_id in _CAP_NOTICES:
        find_period = _capped_period
    elif job_id in _MISSED_DAY_NOTICES:
        find_period = _missed_day
    else:
        return
    try:
        period_key = find_period(manager, job_id, now)
        if period_key is None:
            return
        message = _build_message(manager, job_id, period_key)
        logger.warning(f"補完をやめた期間: job_id={job_id} period_key={period_key}")
        if _send(message):
            _record_sent(manager, job_id, period_key, now)
        else:
            logger.error(f"上限の知らせを送れなかった（次のポーリングで送り直す）: {job_id}")
    except Exception:
        logger.error(f"上限の知らせに失敗（次のポーリングへ進む）: {job_id}", exc_info=True)
