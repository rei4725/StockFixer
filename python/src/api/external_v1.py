"""
外部向け read-only API エンドポイント（R-304）

公開経路:
  GET /api/v1/predictions/top?market=<jp|us>
  GET /api/v1/reports/monthly?month=<YYYY-MM>

認証:
  X-API-Key ヘッダー（環境変数 EXTERNAL_API_KEYS にカンマ区切りで設定）

利用制限:
  1分間に EXTERNAL_API_MAX_RPM リクエスト（デフォルト 60）

公開フィールド定義（内部情報除外方針）:
  予測:  market, symbol, current_price, avg_pred_price, diff_ratio,
         pred_lower_10, pred_upper_90, confluence_score
         除外: model_count, confidence_ratio, model_version, 多ホライズン差分
  月次:  target_month, generated_at, net_return, max_drawdown, sharpe_ratio,
         hit_rate, symbol_count
         除外: avg_slippage（内部執行コスト）, wf_snapshot_file（内部ファイル参照）
"""

import os
import time
from collections import defaultdict
from threading import Lock
from typing import Any

from flask import Flask, Response, jsonify, request

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

_MAX_REQUESTS_PER_MINUTE = int(os.getenv("EXTERNAL_API_MAX_RPM", "60"))

# 予測公開フィールド（内部実装詳細・秘密係数を除外）
# 除外: model_count, confidence_ratio, model_version, avg_pred_price_3d/5d/10d, diff_ratio_3d/5d/10d
_PUBLIC_PREDICTION_FIELDS = (
    "market",
    "symbol",
    "current_price",
    "avg_pred_price",
    "diff_ratio",
    "pred_lower_10",
    "pred_upper_90",
    "confluence_score",
)

# 月次レポート公開フィールド（内部執行コスト・内部ファイル参照を除外）
# 除外: avg_slippage（ペーパー執行コスト詳細）, wf_snapshot_file（内部ファイル名）
_PUBLIC_MONTHLY_FIELDS = (
    "target_month",
    "generated_at",
    "net_return",
    "max_drawdown",
    "sharpe_ratio",
    "hit_rate",
    "symbol_count",
)

# ---------------------------------------------------------------------------
# APIキー認証
# ---------------------------------------------------------------------------


def _load_api_keys() -> frozenset[str]:
    """環境変数 EXTERNAL_API_KEYS からAPIキーセットを返す。"""
    raw = os.getenv("EXTERNAL_API_KEYS", "")
    return frozenset(k.strip() for k in raw.split(",") if k.strip())


def _check_api_key(key: str | None) -> bool:
    if not key:
        return False
    return key in _load_api_keys()


# ---------------------------------------------------------------------------
# レートリミッター（スライディングウィンドウ、インメモリ）
# ---------------------------------------------------------------------------

_rate_lock = Lock()
_request_log: dict[str, list[float]] = defaultdict(list)


def _is_rate_limited(api_key: str) -> bool:
    """1分間のリクエスト数が上限を超えているか確認し、超えていなければ記録する。"""
    now = time.monotonic()
    window_start = now - 60.0
    with _rate_lock:
        timestamps = _request_log[api_key]
        _request_log[api_key] = [t for t in timestamps if t > window_start]
        if len(_request_log[api_key]) >= _MAX_REQUESTS_PER_MINUTE:
            return True
        _request_log[api_key].append(now)
        return False


# ---------------------------------------------------------------------------
# データ変換（内部情報を除外）
# ---------------------------------------------------------------------------


def _to_public_dict(result: Any) -> dict:
    """PredictionResult から公開フィールドのみを抽出した dict を返す。"""
    return {field: getattr(result, field, None) for field in _PUBLIC_PREDICTION_FIELDS}


def _to_public_monthly_dict(summary: Any) -> dict:
    """MonthlyReportSummary から公開フィールドのみを抽出した dict を返す。"""
    return {field: getattr(summary, field, None) for field in _PUBLIC_MONTHLY_FIELDS}


# ---------------------------------------------------------------------------
# ルート登録
# ---------------------------------------------------------------------------


def register_routes(flask_app: Flask) -> None:
    """Flask アプリに /api/v1/* ルートを登録する（冪等）。"""
    if "external_v1_predictions_top" in flask_app.view_functions:
        return

    @flask_app.route("/api/v1/predictions/top")
    def external_v1_predictions_top() -> tuple[Response, int]:
        """最新予測ランキング（Top/Worst）を外部向けに返す。

        内部情報（注文詳細・係数・モデルバージョン）を除外した公開フィールドのみを返す。
        """
        api_key = request.headers.get("X-API-Key")
        if not _check_api_key(api_key):
            return jsonify({"error": "Unauthorized"}), 401

        if _is_rate_limited(api_key):  # type: ignore[arg-type]
            return jsonify({"error": "Too Many Requests"}), 429

        market_filter = request.args.get("market")

        try:
            from src.reporting.query_service import get_latest_market_prediction_snapshots

            predicted_at, snapshots = get_latest_market_prediction_snapshots()
            if not snapshots:
                return jsonify({"predicted_at": None, "markets": []}), 200

            markets_payload = []
            for snap in snapshots:
                if market_filter and snap.market != market_filter:
                    continue
                markets_payload.append(
                    {
                        "market": snap.market,
                        "top": [_to_public_dict(r) for r in snap.top_results],
                        "worst": [_to_public_dict(r) for r in snap.worst_results],
                    }
                )

            return jsonify({"predicted_at": predicted_at, "markets": markets_payload}), 200

        except Exception:
            logger.error("外部APIエンドポイントで例外発生", exc_info=True)
            return jsonify({"error": "Internal Server Error"}), 500

    logger.info("/api/v1/predictions/top ルートを登録しました")

    @flask_app.route("/api/v1/reports/monthly")
    def external_v1_reports_monthly() -> tuple[Response, int]:
        """月次KPIサマリーを外部向けに返す。

        内部執行コスト・内部ファイル参照を除外した公開フィールドのみを返す。
        ?month=YYYY-MM で対象月を指定可（省略時は当月）。
        """
        api_key = request.headers.get("X-API-Key")
        if not _check_api_key(api_key):
            return jsonify({"error": "Unauthorized"}), 401

        if _is_rate_limited(api_key):  # type: ignore[arg-type]
            return jsonify({"error": "Too Many Requests"}), 429

        target_month = request.args.get("month")

        try:
            from src.reporting.query_service import get_monthly_report_summary

            summary = get_monthly_report_summary(target_month=target_month)
            return jsonify(_to_public_monthly_dict(summary)), 200

        except Exception:
            logger.error("月次レポートAPIエンドポイントで例外発生", exc_info=True)
            return jsonify({"error": "Internal Server Error"}), 500

    logger.info("/api/v1/reports/monthly ルートを登録しました")
