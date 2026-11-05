"""スケジューラ状態ファイルから日次パイプラインの連続失敗回数を数える（NF-303-1 の部品）。"""

from __future__ import annotations

import json
import os

# スケジューラ日次パイプラインのジョブID
_DAILY_PIPELINE_JOB_ID = "daily_pipeline"


class SchedulerStateUnreadableError(Exception):
    """スケジューラ状態ファイルが在るのに読めない（#924）。"""


def _count_consecutive_pipeline_failures(state_file_path: str | None = None) -> int:
    """scheduler_queue_state.json から daily_pipeline の連続失敗回数を返す。

    ファイルが無いだけ（初回の起動）なら 0。在るのに読めない、または壊れて
    退避された跡（.corrupt-*）だけが残っているときは SchedulerStateUnreadableError
    を投げる（0 回と数えて黙らないため、#924）。
    """
    if state_file_path is None:
        from src.utils.data_path_utils import get_results_dir

        state_file_path = os.path.join(get_results_dir(), "scheduler_queue_state.json")

    if not os.path.exists(state_file_path):
        quarantined = _find_quarantined_state(state_file_path)
        if quarantined:
            raise SchedulerStateUnreadableError(
                f"状態ファイルが退避されたまま本体が無い: {quarantined}"
            )
        return 0

    try:
        with open(state_file_path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SchedulerStateUnreadableError(f"状態ファイルが読めない: {exc}") from exc
    if not isinstance(state, dict):
        raise SchedulerStateUnreadableError(
            f"状態ファイルが読めない: 最上位が dict ではない: {type(state).__name__}"
        )

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


def _find_quarantined_state(state_file_path: str) -> str | None:
    """state_file_path の隣に退避された状態ファイル（.corrupt-*）の最新の名前を返す。"""
    state_dir = os.path.dirname(os.path.abspath(state_file_path))
    prefix = f"{os.path.basename(state_file_path)}.corrupt-"
    try:
        names = [name for name in os.listdir(state_dir) if name.startswith(prefix)]
    except OSError:
        return None
    return max(names) if names else None
