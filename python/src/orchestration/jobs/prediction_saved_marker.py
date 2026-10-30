"""日次の道の [3/6] の production の予想の保存の終わりの印（#1394）。

[3/6] の始まりで印を消し、保存し終えたら JST の今日の日付を残す。8:50 の自動発注は
印が今日でなければ（[3/6] がまだ・保存の途中・落ちた）新規を止める。印はアプリが
起き直しても残るよう results/ のファイルに書く。
"""

import os

from src.orchestration.jobs.common import today_jst_key
from src.utils.data_path_utils import ensure_dir, get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MARKER_NAME = "production_prediction_saved_day.txt"


def _marker_path() -> str:
    return os.path.join(get_results_dir(), _MARKER_NAME)


def clear_prediction_saved_marker() -> None:
    """[3/6] の始まりで印を消す。消せなくても [3/6] は止めない。"""
    try:
        if os.path.exists(_marker_path()):
            os.remove(_marker_path())
    except OSError:
        logger.error("[3/6] 予想の保存の終わりの印を消せない", exc_info=True)


def mark_prediction_saved() -> None:
    """[3/6] の保存し終えた後に、JST の今日の日付を印に残す。"""
    try:
        ensure_dir(get_results_dir())
        with open(_marker_path(), "w", encoding="utf-8") as f:
            f.write(today_jst_key())
    except OSError:
        logger.error("[3/6] 予想の保存の終わりの印を書けない", exc_info=True)


def read_prediction_saved_day() -> str:
    """印の JST の日付（YYYYMMDD）を返す。無い・読めなければ空（新規を止める側）。"""
    try:
        with open(_marker_path(), encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""
    except OSError:
        logger.error("予想の保存の終わりの印を読めない", exc_info=True)
        return ""
