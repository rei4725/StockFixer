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

# health 用 DB チェックのロック取得待ち上限（秒）。
# Docker HEALTHCHECK の HTTP タイムアウト（8秒）内に必ず応答を返せるよう、
# 既定の 120 秒ではなく短い値でロック待ちを打ち切る（#550）。
_DB_CHECK_LOCK_TIMEOUT = 2.0

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

    ロック待ちは _DB_CHECK_LOCK_TIMEOUT 秒で打ち切り "busy" を返す（#550）。
    busy は「別処理（日次パイプライン等）が DB を使用中 = プロセスは生きている」
    ことを意味し、異常ではない。DB 破損のような真の異常はロック取得後の
    接続失敗として "error" になる。
    """
    try:
        from src.utils.db._connection import DbLockTimeoutError, _db_connection

        try:
            with _db_connection(lock_timeout=_DB_CHECK_LOCK_TIMEOUT) as con:
                con.execute("SELECT 1").fetchone()
        except DbLockTimeoutError:
            return "busy", None
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
        from src.utils.db._connection import DbLockTimeoutError, _db_connection

        try:
            with _db_connection(lock_timeout=_DB_CHECK_LOCK_TIMEOUT) as con:
                row = con.execute("SELECT MAX(predicted_at) FROM prediction_results").fetchone()
        except DbLockTimeoutError:
            # 別処理が DB 使用中。異常ではないので warning にしない
            return None
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

    # busy = 別処理（バッチ等）が DB 使用中で、プロセスとしては健全（#550）
    overall = "ok" if db_status in ("ok", "busy") and not scheduler_stale else "degraded"

    payload: dict[str, Any] = {
        "status": overall,
        "db": db_status if db_error is None else f"error: {db_error}",
        "scheduler_last_runs": scheduler_runs,
        "last_prediction_at": _get_last_prediction_at(),
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
