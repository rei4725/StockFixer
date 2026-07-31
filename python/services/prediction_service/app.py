"""予測配信マイクロサービスの FastAPI アプリ。

起動:
    cd python
    uvicorn services.prediction_service.app:app --host 0.0.0.0 --port 5200
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from services.prediction_service.inference import load_model, run_inference
from services.prediction_service.types import HealthResponse, PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

# ヘルスチェックで存在確認する既定のモデル名
_DEFAULT_MODEL_NAMES = ("UnifiedStockXGBoost", "UnifiedStockLightGBM")

app = FastAPI(
    title="StockFixer Prediction Service",
    description="特徴量と現在価格を受け取り、モデル推論とアンサンブルのみを行う計算サービス",
    version="1.0.0",
)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """推論を実行して予測価格を返す。

    全モデルの推論が失敗した場合も 200 を返し、model_count=0 で表現する。
    """
    return run_inference(request)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """モデルがロード可能かを返す。

    1つもロードできない場合は status="degraded" とする（HTTP は 200 のまま。
    サービスプロセス自体は生きているため）。
    """
    loaded = [name for name in _DEFAULT_MODEL_NAMES if load_model(name) is not None]
    return HealthResponse(status="ok" if loaded else "degraded", loaded_models=loaded)
