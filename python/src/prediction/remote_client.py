"""予測配信マイクロサービスの HTTP クライアント（#予測配信サービス フェーズ1）。

環境変数 PREDICTION_SERVICE_URL が設定されているときだけサービスを呼ぶ。
未設定時・呼び出し失敗時はいずれも None を返し、呼び出し側は従来の
インプロセス推論へフォールバックする。

このモジュールは services/ を import しない。本体とサービスは独立して
デプロイされるため、HTTP のみが両者の契約である。
"""

from __future__ import annotations

import os
from typing import Optional

import requests

from src.prediction.types import PredictionResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

_REQUEST_TIMEOUT_SECONDS = 10.0


def get_service_url() -> Optional[str]:
    """PREDICTION_SERVICE_URL を返す。未設定・空文字なら None。"""
    url = os.getenv("PREDICTION_SERVICE_URL", "").strip()
    return url or None


def predict_via_service(
    market: str,
    symbol: str,
    current_price: float,
    features: dict[str, float],
    model_types: list[str],
    model_weights: list[float],
) -> Optional[PredictionResult]:
    """推論サービスへ HTTP で問い合わせる。

    Returns:
        成功時は PredictionResult。以下の場合はいずれも None を返し、
        呼び出し側はインプロセス推論へフォールバックする:
          - PREDICTION_SERVICE_URL 未設定
          - 接続不可・タイムアウト・5xx
          - 応答が想定した形式でない
          - model_count が 0（サービス側で全モデル失敗＝予測なし）
    """
    base_url = get_service_url()
    if base_url is None:
        return None

    payload = {
        "market": market,
        "symbol": symbol,
        "current_price": current_price,
        "features": features,
        "model_types": model_types,
        "model_weights": model_weights,
    }

    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/predict",
            json=payload,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as exc:
        logger.warning(
            "推論サービス呼び出し失敗（インプロセス推論にフォールバック）: "
            "market=%s symbol=%s error=%s",
            market,
            symbol,
            exc,
        )
        return None

    if response.status_code != 200:
        logger.warning(
            "推論サービスがエラー応答（インプロセス推論にフォールバック）: "
            "market=%s symbol=%s status=%s",
            market,
            symbol,
            response.status_code,
        )
        return None

    try:
        body = response.json()
        model_count = int(body["model_count"])
        if model_count == 0:
            return None
        return PredictionResult(
            market=str(body["market"]),
            symbol=str(body["symbol"]),
            current_price=float(body["current_price"]),
            avg_pred_price=float(body["avg_pred_price"]),
            diff_ratio=float(body["diff_ratio"]),
            model_count=model_count,
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "推論サービスの応答を解釈できません（インプロセス推論にフォールバック）: "
            "market=%s symbol=%s error=%s",
            market,
            symbol,
            exc,
        )
        return None
