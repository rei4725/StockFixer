"""
Flask /health エンドポイント（NF-301）

DB接続・スケジューラ最終実行時刻・直近予測実行時刻を JSON で返す。
Docker HEALTHCHECK から叩かれることを想定。

起動:
    port = int(os.getenv("HEALTH_PORT", "5100"))
    start_health_server(port=port)   # daemon thread として起動
"""

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

from flask import Flask, Response, jsonify

from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

_SCHEDULER_STALE_SECS = 30 * 60  # 30分

# health 用 DB チェックの接続待ち上限（秒）。
# Docker HEALTHCHECK の HTTP タイムアウト（8秒）内に必ず応答を返せるよう、
# 既定の 30 秒ではなく短い値でプール接続待ちを打ち切る（#550）。
_DB_CHECK_LOCK_TIMEOUT = 2.0

logger = get_logger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------


def _check_db() -> tuple[str, str | None, str | None]:
    """DB接続を確認する。(status, error_msg, last_prediction_at) を返す。

    health サーバは scheduler / bot と同一プロセスで動くため、read-only の別接続
    （get_readonly_connection）は使わず、共有のプールから借用する _db_connection
    経由で読む。

    プールが空でタイムアウトした場合は _DB_CHECK_LOCK_TIMEOUT 秒で打ち切り "busy"
    を返す（#550）。busy は「別処理（日次パイプライン等）が DB を使用中 =
    プロセスは生きている」ことを意味し、異常ではない。DB 破損のような真の異常は
    接続取得後の失敗として "error" になる。

    接続は 1 回だけ張り、疎通確認と last_prediction_at 取得をまとめて行う
    （#553）。
    """
    try:
        from src.utils.db._connection import DbLockTimeoutError, _db_connection

        try:
            with _db_connection(lock_timeout=_DB_CHECK_LOCK_TIMEOUT) as con:
                con.execute("SELECT 1").fetchone()
                last_prediction_at = _query_last_prediction_at(con)
        except DbLockTimeoutError:
            return "busy", None, None
        return "ok", None, last_prediction_at
    except Exception as exc:
        return "error", str(exc), None


def _query_last_prediction_at(con: Any) -> str | None:
    """開いている接続から直近の予測実行時刻を返す。失敗しても健全性判定に影響させない。"""
    try:
        row = con.execute("SELECT MAX(predicted_at) FROM prediction_results").fetchone()
        return row[0] if row and row[0] else None
    except Exception as exc:
        logger.warning("last_prediction_at 取得失敗: %s", exc)
        return None


def _load_scheduler_last_runs() -> dict[str, str | None]:
    """スケジューラ状態ファイルから各ジョブの最終成功実行時刻を返す。"""
    state_path = os.path.join(get_results_dir(), "scheduler_queue_state.json")
    if not os.path.exists(state_path):
        return {}

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

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


# ---------------------------------------------------------------------------
# エンドポイント
# ---------------------------------------------------------------------------


from src.api.external_v1 import register_routes as _register_external_v1  # noqa: E402
from src.api.metrics import register_routes as _register_metrics  # noqa: E402

_register_metrics(app)
_register_external_v1(app)


@app.route("/health")
def health() -> tuple[Response, int]:
    db_status, db_error, last_prediction_at = _check_db()
    scheduler_runs = _load_scheduler_last_runs()

    scheduler_stale = False
    if scheduler_runs:
        valid_runs = [v for v in scheduler_runs.values() if v is not None]
        latest_ts = max(valid_runs) if valid_runs else None
        if latest_ts:
            try:
                last_dt = datetime.fromisoformat(latest_ts)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
                scheduler_stale = elapsed > _SCHEDULER_STALE_SECS
            except ValueError:
                pass

    # busy = 別処理（バッチ等）が DB 使用中で、プロセスとしては健全（#550）
    overall = "ok" if db_status in ("ok", "busy") and not scheduler_stale else "degraded"

    payload: dict[str, Any] = {
        "status": overall,
        "db": db_status if db_error is None else f"error: {db_error}",
        "scheduler_last_runs": scheduler_runs,
        "last_prediction_at": last_prediction_at,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }

    http_status = 200 if db_status in ("ok", "busy") else 503
    return jsonify(payload), http_status


# ---------------------------------------------------------------------------
# サーバー起動
# ---------------------------------------------------------------------------


def start_health_server(port: int = 5100) -> None:
    """health サーバーを本番用 WSGI サーバ（waitress）でデーモンスレッド起動する。

    Flask 開発サーバ（werkzeug の app.run）は本番運用に非対応のため、純Python で
    Windows 相性の良い waitress で WSGI アプリ（app）を直接配信する。
    """

    def _run() -> None:
        from waitress import serve  # type: ignore[import-untyped]

        logger.info("Health サーバー起動: port=%d (waitress)", port)
        serve(app, host="0.0.0.0", port=port, threads=4)

    thread = threading.Thread(target=_run, daemon=True, name="health-server")
    thread.start()
