"""
前の起動のプロセスで消えた回（終わりの記録の無い started）の見分け（#946）

始めの記録には起動ごとの印（boot_id）を入れる。今の起動と違う印を持ち、今の起動より前に
始まった終わりの無い started を「消えた回」とする。同じプロセスで動いている回は印が同じで、
今の起動の後に別のプロセスが手で始めた回は始めの時刻が今の起動より後なので、どちらも消えた回と
しない。印の無い前の版の記録も、消えた回としない（今までどおり回数に数える）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

from src.orchestration.scheduler_backfill_cover import has_success_since
from src.orchestration.scheduler_cutoff import (
    is_past_recovery_until,
    scheduled_datetime_for_recovery,
)

logger = logging.getLogger("scheduler")

# 前日分の補完が遡る日数。run_scheduler の補完のポーリングと同じ値（#1783）
BACKFILL_LOOKBACK_DAYS = 2

# 消えた回を回数に数えずに補完する、1期間あたりの上限。2回続けて消えたら止める（C-014）。
# ジョブの設定の max_interrupted_reruns_per_period で、そのジョブだけ増やせる（#1522）
MAX_FORGIVEN_INTERRUPTIONS_PER_PERIOD = 1


@dataclass(frozen=True)
class ExecutionCount:
    """1つの期間の実行回数と、その中の消えた回"""

    count: int
    interrupted_run_ids: list[str] = field(default_factory=list)


def _started_before(event: dict[str, Any], booted_at: datetime) -> bool:
    started_at = event.get("started_at")
    if not isinstance(started_at, str):
        return False
    try:
        return datetime.fromisoformat(started_at) < booted_at
    except ValueError:
        return False


def count_executions(
    events: list[dict[str, Any]],
    job_id: str,
    period_key: str,
    *,
    boot_id: str,
    booted_at: datetime,
    rerun_if_interrupted: bool,
    max_forgiven: int = MAX_FORGIVEN_INTERRUPTIONS_PER_PERIOD,
) -> ExecutionCount:
    """success・error の記録に、終わりの記録が無い started（途中で落ちた実行）を足して数える。

    rerun_if_interrupted のジョブは、消えた回を1期間に max_forgiven 回まで回数から除く。
    """
    count = 0
    started_events: dict[str, dict[str, Any]] = {}
    finished_run_ids: set[str] = set()
    for event in events:
        if event.get("job_id") != job_id or event.get("period_key") != period_key:
            continue
        status = event.get("status")
        if status in {"success", "error"}:
            count += 1
            if event.get("run_id"):
                finished_run_ids.add(event["run_id"])
        elif status == "started" and event.get("run_id"):
            started_events.setdefault(event["run_id"], event)

    interrupted: list[str] = []
    for run_id, event in started_events.items():
        if run_id in finished_run_ids:
            continue
        count += 1
        event_boot_id = event.get("boot_id")
        if event_boot_id and event_boot_id != boot_id and _started_before(event, booted_at):
            interrupted.append(run_id)

    if rerun_if_interrupted:
        count -= min(len(interrupted), max_forgiven)
    return ExecutionCount(count=count, interrupted_run_ids=interrupted)


def is_backfill_daily(config: dict[str, Any]) -> bool:
    """前日分を補完するジョブなら True。daily で backfill の印を持つジョブだけ（#825）。"""
    return config.get("period", "daily") == "daily" and bool(config.get("backfill", False))


@dataclass(frozen=True)
class PeriodCalendar:
    """1つのジョブの曜日・期間の鍵・回数の数え方。期間の判定の関数に渡す"""

    weekdays: set[int]
    is_scheduled_day: Callable[[datetime], bool]
    period_key_at: Callable[[datetime], str]
    count_at: Callable[[str], int]


def latest_missed_past_period(
    config: dict[str, Any],
    events: list[dict[str, Any]],
    job_id: str,
    now: datetime,
    lookback_days: int,
    calendar: PeriodCalendar,
) -> str | None:
    """前日分の補完の対象にする、抜けた最も新しい1日の期間を返す（#1719）。

    後の日の success で覆われた日と、回数が上限に届いた日は飛ばす。
    is_backfill_daily のジョブだけを渡す。
    """
    max_executions = config.get("max_executions_per_period", 1)
    for days_back in range(1, lookback_days + 1):
        past_dt = now - timedelta(days=days_back)
        if not calendar.is_scheduled_day(past_dt):
            continue
        past_period_key = calendar.period_key_at(past_dt)
        if has_success_since(events, job_id, past_period_key):
            continue
        if calendar.count_at(past_period_key) >= max_executions:
            continue
        return past_period_key
    return None


def period_still_recoverable(
    config: dict[str, Any],
    events: list[dict[str, Any]],
    job_id: str,
    period_key: str,
    now: datetime,
    calendar: PeriodCalendar,
) -> bool:
    """period_key の期間を、now から後の補完がまだ動かしうるなら True（#1783）。

    今の期間は should_recover の日と締めの判定に、前の期間は get_missed_past_periods の
    条件に合わせる。補完のポーリングに乗らない cron 式のジョブは、今までどおり True とする。
    """
    if not isinstance(config.get("hour"), int) or not isinstance(config.get("minute"), int):
        return True
    if period_key == calendar.period_key_at(now):
        scheduled_dt = scheduled_datetime_for_recovery(config, now, calendar.weekdays)
        return scheduled_dt is not None and not is_past_recovery_until(config, now)
    if not is_backfill_daily(config):
        return False
    latest = latest_missed_past_period(
        config, events, job_id, now, BACKFILL_LOOKBACK_DAYS, calendar
    )
    return latest == period_key


def warn_interrupted_runs(job_id: str, period_key: str, run_ids: list[str], label: str) -> None:
    """消えた回を1つずつ WARNING に書く。"""
    for run_id in run_ids:
        logger.warning(
            f"前の起動のプロセスで消えた回を見つけました: job_id={job_id} "
            f"period_key={period_key} run_id={run_id} "
            f"補完の対象={label}"
        )


def interrupted_recovery_label(
    rerun_if_interrupted: bool,
    max_forgiven: int,
    max_executions: int,
    counted: ExecutionCount,
    recoverable: bool = True,
) -> str:
    """消えた回の警告に書く、補完のされ方。

    rerun_if_interrupted の無いジョブも、消えた回を上限に数えたうえで残りの回で補完する。
    「なし」とだけ書くと、やり直さないと読めるため、上限に数えることを書く（#1769）。
    counted は数えたあとの回数とその期間の消えた回。回数が上限に届いていれば
    補完しないので、補完すると書かない（#1777）。recoverable は period_still_recoverable の
    結果で、補完の日を過ぎた期間にも補完すると書かない（#1783）。判定は should_recover と
    get_missed_past_periods が行う。
    """
    if counted.count >= max_executions:
        if rerun_if_interrupted and len(counted.interrupted_run_ids) > max_forgiven:
            return (
                f"なし（数えない枠の{max_forgiven}回を超えた消えた回を数え、"
                f"上限の{max_executions}回に届いたので補完しない）"
            )
        return f"なし（上限の{max_executions}回に届いたので補完しない）"
    if not recoverable:
        return "なし（補完の日を過ぎたので補完しない）"
    if rerun_if_interrupted:
        return f"あり（1期間に{max_forgiven}回まで）"
    return f"上限の{max_executions}回に数えて補完（#1769）"
