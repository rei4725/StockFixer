"""
Prometheus /metrics エンドポイント（NF-metrics）

公開メトリクス:
  pipeline_duration_seconds{pipeline}    Histogram  パイプライン実行時間
  pipeline_runs_total{pipeline,status}   Counter    パイプライン実行回数
  duckdb_query_duration_seconds          Histogram  DuckDB クエリ実行時間

起動:
    from src.api.metrics import register_routes
    register_routes(app)   # Flask app に /metrics ルートを追加
"""

import contextlib
import time
from typing import Generator

from flask import Flask, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# メトリクス定義
# ---------------------------------------------------------------------------

pipeline_duration = Histogram(
    "pipeline_duration_seconds",
    "Pipeline execution duration in seconds",
    labelnames=["pipeline"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800),
)

pipeline_runs_total = Counter(
    "pipeline_runs_total",
    "Total number of pipeline runs",
    labelnames=["pipeline", "status"],
)

duckdb_query_duration = Histogram(
    "duckdb_query_duration_seconds",
    "DuckDB query execution duration in seconds",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
)


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def track_pipeline(name: str) -> Generator[None, None, None]:
    """パイプライン実行時間と成否を Prometheus メトリクスに記録するコンテキストマネージャ。"""
    start = time.perf_counter()
    try:
        yield
        elapsed = time.perf_counter() - start
        pipeline_duration.labels(pipeline=name).observe(elapsed)
        pipeline_runs_total.labels(pipeline=name, status="success").inc()
    except Exception:
        elapsed = time.perf_counter() - start
        pipeline_duration.labels(pipeline=name).observe(elapsed)
        pipeline_runs_total.labels(pipeline=name, status="fail").inc()
        raise


# ---------------------------------------------------------------------------
# ルート登録
# ---------------------------------------------------------------------------


def register_routes(flask_app: Flask) -> None:
    """Flask アプリに /metrics ルートを登録する（冪等）。"""
    if "metrics_endpoint" in flask_app.view_functions:
        return

    @flask_app.route("/metrics")
    def metrics_endpoint() -> Response:
        data = generate_latest()
        return Response(data, status=200, mimetype=CONTENT_TYPE_LATEST)

    logger.info("/metrics ルートを登録しました")
