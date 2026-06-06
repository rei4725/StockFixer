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

logger = get_logger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------


def _check_db() -> tuple[str, str | None]:
    """DB接続を確認する。(status, error_msg) を返す。

    health サーバは scheduler / bot と同一プロセスで動くため、read-only の別接続
    （get_readonly_connection）を開くと read-write 接続と設定が衝突する
    （DuckDB は同一プロセスで同一ファイルへ異なる設定の接続を許さない）。
    そのため共有の _db_connection（FileLock 直列化・設定統一）経由で読む。
    """
    try:
        from src.utils.db._connection import _db_connection

        with _db_connection() as con:
            con.execute("SELECT 1").fetchone()
        return "ok", None
    except Exception as exc:
        return "error", str(exc)


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


def _get_last_prediction_at() -> str | None:
    """prediction_results テーブルから直近の予測実行時刻を返す。"""
    try:
        # 同一プロセス内のため共有接続を使う（read-only 別接続だと設定衝突する）
        from src.utils.db._connection import _db_connection

        with _db_connection() as con:
            row = con.execute("SELECT MAX(predicted_at) FROM prediction_results").fetchone()
        return row[0] if row and row[0] else None
    except Exception as exc:
        logger.warning("last_prediction_at 取得失敗: %s", exc)
        return None


# ---------------------------------------------------------------------------
# エンドポイント
# ---------------------------------------------------------------------------


from src.api.external_v1 import register_routes as _register_external_v1  # noqa: E402
from src.api.metrics import register_routes as _register_metrics  # noqa: E402

_register_metrics(app)
_register_external_v1(app)


@app.route("/health")
def health() -> tuple[Response, int]:
    db_status, db_error = _check_db()
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

    overall = "ok" if db_status == "ok" and not scheduler_stale else "degraded"

    payload: dict[str, Any] = {
        "status": overall,
        "db": db_status if db_error is None else f"error: {db_error}",
        "scheduler_last_runs": scheduler_runs,
        "last_prediction_at": _get_last_prediction_at(),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }

    http_status = 200 if db_status == "ok" else 503
    return jsonify(payload), http_status


# ---------------------------------------------------------------------------
# サーバー起動
# ---------------------------------------------------------------------------


def start_health_server(port: int = 5100) -> None:
    """Flask health サーバーをデーモンスレッドで起動する。"""

    def _run() -> None:
        logger.info("Health サーバー起動: port=%d", port)
        app.run(host="0.0.0.0", port=port, use_reloader=False, threaded=True)

    thread = threading.Thread(target=_run, daemon=True, name="health-server")
    thread.start()
