"""スケジューラ状態ファイルから日次パイプラインの連続失敗回数を数える（NF-303-1 の部品）。"""

from __future__ import annotations

import os

from src.utils.scheduler_state_read import (
    SCHEDULER_STATE_FILE_NAME,
    SchedulerStateUnreadableError,
    read_scheduler_state,
)

__all__ = ["SchedulerStateUnreadableError"]

# スケジューラ日次パイプラインのジョブID
_DAILY_PIPELINE_JOB_ID = "daily_pipeline"


def _count_consecutive_pipeline_failures(state_file_path: str | None = None) -> int:
    """scheduler_queue_state.json から daily_pipeline の連続失敗回数を返す。

    ファイルが無いだけ（初回の起動）なら 0。在るのに読めない、または壊れて
    退避された跡（.corrupt-*）だけが残っているときは SchedulerStateUnreadableError
    を投げる（0 回と数えて黙らないため、#924）。
    """
    if state_file_path is None:
        from src.utils.data_path_utils import get_results_dir

        state_file_path = os.path.join(get_results_dir(), SCHEDULER_STATE_FILE_NAME)

    state = read_scheduler_state(state_file_path)
    if state is None:
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
