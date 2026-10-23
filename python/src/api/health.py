"""
Flask /health エンドポイント（NF-301）

DB接続・スケジューラ最終実行時刻・直近予測実行時刻を JSON で返す。
Docker HEALTHCHECK から叩かれることを想定。

起動:
    port = int(os.getenv("HEALTH_PORT", "5100"))
    start_health_server(port=port)   # daemon thread として起動
"""

import threading
from datetime import datetime, timezone
from typing import Any

from flask import Flask, Response, jsonify

from src.utils.data_path_utils import get_results_dir
from src.utils.health_check import (
    SchedulerStateUnreadableError,
    check_db,
    is_scheduler_stale,
    load_scheduler_last_runs,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


# ---------------------------------------------------------------------------
# ヘルパー（判定の部品は src/utils/health_check.py。日次の道と共有する、#940）
# ---------------------------------------------------------------------------


def _check_db() -> tuple[str, str | None, str | None]:
    """DB接続を確認する。(status, error_msg, last_prediction_at) を返す。"""
    return check_db()


def _load_scheduler_last_runs() -> dict[str, str | None]:
    """スケジューラ状態ファイルから各ジョブの最終成功実行時刻を返す。

    在るのに読めないときは SchedulerStateUnreadableError を投げる（#924）。
    """
    return load_scheduler_last_runs(get_results_dir())


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
    scheduler_state_error: str | None = None
    try:
        scheduler_runs = _load_scheduler_last_runs()
    except SchedulerStateUnreadableError as exc:
        logger.error("スケジューラ状態ファイルが読めない: %s", exc)
        scheduler_runs = {}
        scheduler_state_error = str(exc)

    scheduler_stale = is_scheduler_stale(scheduler_runs)

    # busy = 別処理（バッチ等）が DB 使用中で、プロセスとしては健全（#550）
    healthy = db_status in ("ok", "busy") and not scheduler_stale
    overall = "ok" if healthy and scheduler_state_error is None else "degraded"

    payload: dict[str, Any] = {
        "status": overall,
        "db": db_status if db_error is None else f"error: {db_error}",
        "scheduler_last_runs": scheduler_runs,
        "last_prediction_at": last_prediction_at,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    if scheduler_state_error is not None:
        payload["scheduler_state_error"] = scheduler_state_error

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
