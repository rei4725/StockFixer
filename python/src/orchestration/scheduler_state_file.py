"""
スケジューラの状態ファイルの読み・退避・置き換えの書き込み

ジョブの判定の中身は持たず、ファイルの扱いだけを行う。
ロックは呼ぶ側（SchedulerQueueManager）が取る（#826・#1850）。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from src.utils.data_path_utils import ensure_dir

logger = logging.getLogger("scheduler")


@dataclass(frozen=True)
class StateFileRead:
    """状態ファイルを読んだ結果。

    state が None なら読めなかった。corrupt_detail があれば、壊れていて退避すべき。
    """

    state: Optional[dict[str, Any]]
    corrupt_detail: Optional[str] = None


def read_state_file(path: str) -> StateFileRead:
    """状態ファイルを読む。無ければ空の履歴、壊れていれば退避の理由を返す。"""
    if not os.path.exists(path):
        return StateFileRead({"events": []})

    try:
        with open(path, "r", encoding="utf-8") as file:
            state = json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return StateFileRead(None, str(exc))
    except OSError:
        logger.error("スケジューラ状態ファイル読込失敗", exc_info=True)
        return StateFileRead(None)

    if not isinstance(state, dict):
        return StateFileRead(None, f"最上位が dict ではない: {type(state).__name__}")
    events = state.get("events")
    if not isinstance(events, list):
        state["events"] = []
    return StateFileRead(state)


def quarantine_state_file(path: str, detail: str) -> Optional[str]:
    """壊れた状態ファイルを隣へ日時つきで退避する。退避先を返し、失敗なら None。"""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    quarantine_path = f"{path}.corrupt-{stamp}"
    try:
        os.replace(path, quarantine_path)
    except OSError:
        logger.error(
            f"壊れたスケジューラ状態ファイルの退避に失敗: {path}",
            exc_info=True,
        )
        return None
    logger.error(
        "スケジューラ状態ファイルが読めないため退避し、補完実行を止めます"
        f"（定時のジョブは続けます）: {detail} 退避先={quarantine_path}"
    )
    return quarantine_path


def write_state_file(path: str, state: dict[str, Any]) -> None:
    """一時ファイルに書いてから置き換える。失敗したら一時ファイルを消して投げ直す。"""
    state_dir = os.path.dirname(os.path.abspath(path))
    ensure_dir(state_dir)
    fd, tmp_path = tempfile.mkstemp(
        dir=state_dir, prefix=f"{os.path.basename(path)}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
