"""予測配信サービスのリクエスト/レスポンス型のテスト。"""

import pytest
from pydantic import ValidationError
from services.prediction_service.types import HealthResponse, PredictRequest, PredictResponse


def _valid_request_kwargs() -> dict:
    return {
        "market": "jp",
        "symbol": "7203",
        "current_price": 2500.0,
        "features": {"Close_lag1": 2480.0, "rsi_lag1": 55.2},
        "model_types": ["UnifiedStockXGBoost", "UnifiedStockLightGBM"],
        "model_weights": [0.6, 0.4],
    }


def test_valid_request_accepted():
    req = PredictRequest(**_valid_request_kwargs())
    assert req.market == "jp"
    assert req.symbol == "7203"
    assert req.current_price == 2500.0
    assert req.features["Close_lag1"] == 2480.0
    assert req.model_types == ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]
    assert req.model_weights == [0.6, 0.4]


def test_weights_length_mismatch_rejected():
    """model_weights の要素数が model_types と一致しない場合は弾く。"""
    kwargs = _valid_request_kwargs()
    kwargs["model_weights"] = [1.0]
    with pytest.raises(ValidationError):
        PredictRequest(**kwargs)


def test_empty_model_types_rejected():
    kwargs = _valid_request_kwargs()
    kwargs["model_types"] = []
    kwargs["model_weights"] = []
    with pytest.raises(ValidationError):
        PredictRequest(**kwargs)


def test_response_roundtrip():
    resp = PredictResponse(
        market="jp",
        symbol="7203",
        current_price=2500.0,
        avg_pred_price=2537.5,
        diff_ratio=0.015,
        model_count=2,
        used_models=["UnifiedStockXGBoost", "UnifiedStockLightGBM"],
    )
    payload = resp.model_dump()
    assert payload["avg_pred_price"] == 2537.5
    assert payload["model_count"] == 2


def test_health_response():
    health = HealthResponse(status="ok", loaded_models=["UnifiedStockXGBoost"])
    assert health.status == "ok"
    assert health.loaded_models == ["UnifiedStockXGBoost"]
