"""健全性の判定の部品（NF-301・NF-303-4、#940）。

api（/health）と orchestration（日次の道）の両方から使うため、utils に置く。

- /health は DB の疎通・状態ファイルの読めなさ・スケジューラの鮮度（3時間）で判定する
- 日次の道は鮮度を見ない。日次の道が走っていること自体がスケジューラの生存の証のため
"""

import json
import os
from datetime import datetime, timezone
from typing import Any

from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

# 一番頻繁に success を残すジョブ strategy_promotion_check（毎日、2時間ごと）の間隔に、
# 補完の待ち（recovery_delay_minutes の30分）と走る時間の余裕を足した3時間。
# 30分では次の回を待つ間ずっと degraded になっていた（#962）。
# このジョブを止める・間隔を延ばすときは、この値も見直す
SCHEDULER_STALE_SECS = 3 * 60 * 60  # 3時間

# health 用 DB チェックの接続待ち上限（秒）。
# Docker HEALTHCHECK の HTTP タイムアウト（8秒）内に必ず応答を返せるよう、
# 既定の 30 秒ではなく短い値でプール接続待ちを打ち切る（#550）。
DB_CHECK_LOCK_TIMEOUT = 2.0

_STATE_FILE_NAME = "scheduler_queue_state.json"

logger = get_logger(__name__)


def check_db() -> tuple[str, str | None, str | None]:
    """DB接続を確認する。(status, error_msg, last_prediction_at) を返す。

    health サーバは scheduler / bot と同一プロセスで動くため、read-only の別接続
    （get_readonly_connection）は使わず、共有のプールから借用する _db_connection
    経由で読む。

    プールが空でタイムアウトした場合は DB_CHECK_LOCK_TIMEOUT 秒で打ち切り "busy"
    を返す（#550）。busy は「別処理（日次パイプライン等）が DB を使用中 =
    プロセスは生きている」ことを意味し、異常ではない。DB 破損のような真の異常は
    接続取得後の失敗として "error" になる。

    接続は 1 回だけ張り、疎通確認と last_prediction_at 取得をまとめて行う
    （#553）。
    """
    try:
        from src.utils.db._connection import DbLockTimeoutError, _db_connection

        try:
            with _db_connection(lock_timeout=DB_CHECK_LOCK_TIMEOUT) as con:
                con.execute("SELECT 1").fetchone()
                last_prediction_at = query_last_prediction_at(con)
        except DbLockTimeoutError:
            return "busy", None, None
        return "ok", None, last_prediction_at
    except Exception as exc:
        return "error", str(exc), None


def query_last_prediction_at(con: Any) -> str | None:
    """開いている接続から直近の予測実行時刻を返す。失敗しても健全性判定に影響させない。"""
    try:
        row = con.execute("SELECT MAX(predicted_at) FROM prediction_results").fetchone()
        return row[0] if row and row[0] else None
    except Exception as exc:
        logger.warning("last_prediction_at 取得失敗: %s", exc)
        return None


class SchedulerStateUnreadableError(Exception):
    """スケジューラ状態ファイルが在るのに読めない（#924）。"""


def load_scheduler_last_runs(results_dir: str) -> dict[str, str | None]:
    """スケジューラ状態ファイルから各ジョブの最終成功実行時刻を返す。

    ファイルが無いだけ（初回の起動）なら空 dict。在るのに読めない、または
    壊れて退避された跡（.corrupt-*）だけが残っているときは
    SchedulerStateUnreadableError を投げる（履歴なし＝正常と黙らないため、#924）。
    """
    state_path = os.path.join(results_dir, _STATE_FILE_NAME)
    if not os.path.exists(state_path):
        quarantined = find_quarantined_state(results_dir)
        if quarantined:
            raise SchedulerStateUnreadableError(
                f"状態ファイルが退避されたまま本体が無い: {quarantined}"
            )
        return {}

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SchedulerStateUnreadableError(f"状態ファイルが読めない: {exc}") from exc
    if not isinstance(state, dict):
        raise SchedulerStateUnreadableError(
            f"状態ファイルが読めない: 最上位が dict ではない: {type(state).__name__}"
        )

    latest: dict[str, str | None] = {}
    for event in state.get("events", []):
        if event.get("status") != "success":
            continue
        job_id = event.get("job_id")
        finished_at = event.get("finished_at")
        if job_id and finished_at:
            if job_id not in latest or finished_at > latest[job_id]:
                latest[job_id] = finished_at

    return latest


def find_quarantined_state(results_dir: str) -> str | None:
    """退避された状態ファイル（scheduler_queue_state.json.corrupt-*）の最新の名前を返す。"""
    try:
        names = [
            name
            for name in os.listdir(results_dir)
            if name.startswith(f"{_STATE_FILE_NAME}.corrupt-")
        ]
    except OSError:
        return None
    return max(names) if names else None


def is_scheduler_stale(
    scheduler_runs: dict[str, str | None],
    now: datetime | None = None,
    stale_secs: float = SCHEDULER_STALE_SECS,
) -> bool:
    """全ジョブで最も新しい成功から stale_secs を超えたかを返す。履歴なし・読めない時刻は False。"""
    valid_runs = [v for v in scheduler_runs.values() if v is not None]
    latest_ts = max(valid_runs) if valid_runs else None
    if not latest_ts:
        return False
    try:
        last_dt = datetime.fromisoformat(latest_ts)
    except ValueError:
        return False
    if last_dt.tzinfo is None:
        last_dt = last_dt.replace(tzinfo=timezone.utc)
    current = now if now is not None else datetime.now(timezone.utc)
    return (current - last_dt).total_seconds() > stale_secs


def is_degraded_for_daily(results_dir: str | None = None) -> bool:
    """日次の道の degraded の判定。DB の疎通の失敗か、状態ファイルの読めなさで True。

    スケジューラの鮮度は見ない（モジュールの説明を参照）。busy は /health と同じく健全とする。
    """
    db_status, db_error, _ = check_db()
    degraded = db_status == "error"
    if degraded:
        logger.error("日次の健全性判定: DB の疎通に失敗: %s", db_error)
    try:
        load_scheduler_last_runs(results_dir if results_dir is not None else get_results_dir())
    except SchedulerStateUnreadableError as exc:
        logger.error("日次の健全性判定: スケジューラ状態ファイルが読めない: %s", exc)
        degraded = True
    return degraded
