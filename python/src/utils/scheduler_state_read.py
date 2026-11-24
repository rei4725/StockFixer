"""スケジューラ状態ファイルの読み（#924）。健全性の判定と連続失敗のアラートが共に使う。"""

from __future__ import annotations

import json
import os
from typing import Any

SCHEDULER_STATE_FILE_NAME = "scheduler_queue_state.json"


class SchedulerStateUnreadableError(Exception):
    """スケジューラ状態ファイルが在るのに読めない（#924）。"""


def read_scheduler_state(state_file_path: str) -> dict[str, Any] | None:
    """スケジューラ状態ファイルを読む。

    ファイルが無いだけ（初回の起動）なら None。在るのに読めない、または壊れて
    退避された跡（.corrupt-*）だけが残っているときは SchedulerStateUnreadableError
    を投げる（履歴なし＝正常と黙らないため、#924）。
    """
    if not os.path.exists(state_file_path):
        quarantined = find_quarantined_state(state_file_path)
        if quarantined:
            raise SchedulerStateUnreadableError(
                f"状態ファイルが退避されたまま本体が無い: {quarantined}"
            )
        return None

    try:
        with open(state_file_path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SchedulerStateUnreadableError(f"状態ファイルが読めない: {exc}") from exc
    if not isinstance(state, dict):
        raise SchedulerStateUnreadableError(
            f"状態ファイルが読めない: 最上位が dict ではない: {type(state).__name__}"
        )
    return state


def find_quarantined_state(state_file_path: str) -> str | None:
    """state_file_path の隣に退避された状態ファイル（.corrupt-*）の最新の名前を返す。"""
    state_dir = os.path.dirname(os.path.abspath(state_file_path))
    prefix = f"{os.path.basename(state_file_path)}.corrupt-"
    try:
        names = [name for name in os.listdir(state_dir) if name.startswith(prefix)]
    except OSError:
        return None
    return max(names) if names else None
