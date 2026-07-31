"""推論ロジック。

モデルのロード・特徴量アラインメント・推論・重み付きアンサンブルを行う。
DB にも yfinance にもアクセスしない（純粋な計算のみ）。

特徴量アラインメント（モデルの feature_names_in_ に合わせて不足分を 0 で埋め、
順序を揃える）はこのサービスの責務である。モデルファイルを持ち、期待される
特徴量名を知っているのがサービス側だからである。
"""

from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Any

import joblib
import pandas as pd
from services.prediction_service.types import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

# モデルディレクトリ。本体の python/models/unified/ をマウントして共有する。
MODEL_DIR = os.getenv("PREDICTION_MODEL_DIR", "/app/models/unified")

_MODEL_CACHE: dict[str, Any] = {}
_CACHE_LOCK = Lock()


def clear_model_cache() -> None:
    """モデルキャッシュを空にする（テスト用）。"""
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()


def load_model(model_name: str) -> Any | None:
    """モデルをロードしてキャッシュする。見つからない場合は None を返す。"""
    with _CACHE_LOCK:
        if model_name in _MODEL_CACHE:
            return _MODEL_CACHE[model_name]

        path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
        try:
            model = joblib.load(path)
        except FileNotFoundError:
            logger.warning("モデルファイルが見つかりません: %s", path)
            model = None
        except Exception:
            logger.error("モデルロード失敗: %s", path, exc_info=True)
            model = None

        _MODEL_CACHE[model_name] = model
        return model


def _align_features(features: dict[str, float], model: Any) -> pd.DataFrame:
    """モデルが期待する特徴量に合わせて DataFrame を組み立てる。

    不足している特徴量は 0 で埋め、期待される順序に並べ替える。
    モデルが feature_names_in_ を持たない場合は与えられた特徴量をそのまま使う。
    """
    df = pd.DataFrame([features])

    inner = getattr(model, "model", None)
    expected = getattr(inner, "feature_names_in_", None)
    if expected is None:
        return df

    expected_list = list(expected)
    for feat in expected_list:
        if feat not in df.columns:
            df[feat] = 0
    return df[expected_list]


def _extract_prediction(pred: Any) -> float:
    """モデルの返り値（Series / list / スカラー）から float を取り出す。"""
    if isinstance(pred, pd.Series):
        return float(pred.iloc[-1])
    if isinstance(pred, (list, tuple)):
        return float(pred[-1])
    return float(pred)


def run_inference(request: PredictRequest) -> PredictResponse:
    """推論を実行し、重み付きアンサンブルの結果を返す。

    全モデルが失敗した場合も例外にせず、model_count=0 のレスポンスを返す
    （本体側はこれを「予測なし」として扱う）。
    一部のモデルのみ成功した場合、重みは成功分だけで再正規化する。
    """
    pred_prices: list[float] = []
    used_weights: list[float] = []
    used_models: list[str] = []

    for model_name, weight in zip(request.model_types, request.model_weights):
        model = load_model(model_name)
        if model is None:
            continue

        try:
            aligned = _align_features(request.features, model)
            pred_return = _extract_prediction(model.predict(aligned))
        except Exception:
            logger.warning(
                "モデル推論スキップ: model=%s market=%s symbol=%s",
                model_name,
                request.market,
                request.symbol,
                exc_info=True,
            )
            continue

        pred_prices.append(request.current_price * (1 + pred_return))
        used_weights.append(weight)
        used_models.append(model_name)

    if not pred_prices:
        return PredictResponse(
            market=request.market,
            symbol=request.symbol,
            current_price=request.current_price,
            avg_pred_price=0.0,
            diff_ratio=0.0,
            model_count=0,
            used_models=[],
        )

    # 成功したモデルのみで重みを再正規化する（合計が1になるよう保つ）
    weight_sum = sum(used_weights)
    if weight_sum > 0:
        normalized = [w / weight_sum for w in used_weights]
    else:
        normalized = [1.0 / len(pred_prices)] * len(pred_prices)

    avg_pred_price = sum(p * w for p, w in zip(pred_prices, normalized))
    diff_ratio = (avg_pred_price - request.current_price) / request.current_price

    return PredictResponse(
        market=request.market,
        symbol=request.symbol,
        current_price=request.current_price,
        avg_pred_price=float(avg_pred_price),
        diff_ratio=float(diff_ratio),
        model_count=len(pred_prices),
        used_models=used_models,
    )
