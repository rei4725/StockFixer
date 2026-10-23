"""
前の起動のプロセスで消えた回（終わりの記録の無い started）の見分け（#946）

始めの記録には起動ごとの印（boot_id）を入れる。今の起動と違う印を持ち、今の起動より前に
始まった終わりの無い started を「消えた回」とする。同じプロセスで動いている回は印が同じで、
今の起動の後に別のプロセスが手で始めた回は始めの時刻が今の起動より後なので、どちらも消えた回と
しない。印の無い前の版の記録も、消えた回としない（今までどおり回数に数える）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# 消えた回を回数に数えずに補完する、1期間あたりの上限。2回続けて消えたら止める（C-014）
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
) -> ExecutionCount:
    """success・error の記録に、終わりの記録が無い started（途中で落ちた実行）を足して数える。

    rerun_if_interrupted のジョブは、消えた回を1期間に1回まで回数から除く。
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
        count -= min(len(interrupted), MAX_FORGIVEN_INTERRUPTIONS_PER_PERIOD)
    return ExecutionCount(count=count, interrupted_run_ids=interrupted)
