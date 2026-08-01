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
import numpy as np
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


def _as_feature_list(names: Any) -> list[str] | None:
    """特徴量名の並びを list[str] に正規化する。名前として使えなければ None。"""
    if names is None or isinstance(names, (str, bytes)):
        return None
    try:
        items = list(names)
    except TypeError:
        return None
    return [str(name) for name in items] if items else None


def _resolve_expected_features(model: Any) -> list[str] | None:
    """モデルが学習時に使った特徴量名を解決する。解決できなければ None。

    sklearn 互換の ``feature_names_in_`` だけを見ると LightGBM を取りこぼす。
    lightgbm がこの属性を持つようになったのは 4.5.0 からで、それ以前に pickle
    されたモデルは LightGBM 独自の ``feature_name_`` にしか名前を持たない。
    属性が無い場合は AttributeError を送出するプロパティとして実装されている
    ため、``hasattr`` ではなく getattr チェーンで順に解決する（#615）。

    load_model() は joblib.load() の戻り値（生の推定器）を返すので、本体
    （predict_unified.py）と違いラッパーを剥がす必要はない。本体にも同等の
    実装があるが、このサービスは独立したコンテナとして動くため、意図的に
    共有せず重複させている。

    ``booster_.feature_name()`` へはフォールバックしない。numpy 配列で学習した
    モデルでは booster が ``f0`` / ``Column_0`` といった合成名を返し、それを
    本物の特徴量名として扱うと全列 0 埋めのフレームで predict が「成功」して
    しまうためである。
    """
    for attr in ("feature_names_in_", "feature_name_"):
        resolved = _as_feature_list(getattr(model, attr, None))
        if resolved is not None:
            return resolved

    return None


def _align_features(features: dict[str, float], model: Any) -> pd.DataFrame:
    """モデルが期待する特徴量に合わせて DataFrame を組み立てる。

    不足している特徴量は 0 で埋め、期待される順序に並べ替える。
    特徴量名を解決できない場合のみ、与えられた特徴量をそのまま使う。
    """
    df = pd.DataFrame([features])

    expected = _resolve_expected_features(model)
    if expected is None:
        logger.warning("モデルの期待特徴量を解決できずアラインメントをスキップ")
        return df

    for feat in expected:
        if feat not in df.columns:
            df[feat] = 0
    return df[expected]


def _extract_prediction(pred: Any) -> float:
    """モデルの返り値（Series / ndarray / list / スカラー）から float を取り出す。

    実際の sklearn/xgboost 推定器の predict() は ndarray を返すため、これを
    扱えないと実運用では常に失敗する（テストが pd.Series のモックしか使って
    いなかったため見逃されていた）。
    """
    if isinstance(pred, pd.Series):
        return float(pred.iloc[-1])
    if isinstance(pred, (list, tuple, np.ndarray)):
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
            # WARNING では埋もれる。1モデル落ちてもアンサンブルは縮退したまま
            # 予測値を返し続けるため、失敗が見えないと気づけない（#615）。
            logger.error(
                "モデル推論スキップ（アンサンブルが縮退します）: model=%s market=%s symbol=%s",
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
