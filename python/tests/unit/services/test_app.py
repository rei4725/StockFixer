"""FastAPI エンドポイントのテスト（TestClient 経由）。"""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from services.prediction_service.app import app
from services.prediction_service.inference import clear_model_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_model_cache()
    yield
    clear_model_cache()


@pytest.fixture
def client():
    return TestClient(app)


def _make_model(pred_return: float):
    model = MagicMock()
    inner = MagicMock()
    inner.feature_names_in_ = ["a", "b"]
    model.model = inner
    model.predict.return_value = pd.Series([pred_return])
    return model


def _payload(**overrides) -> dict:
    body = {
        "market": "jp",
        "symbol": "7203",
        "current_price": 1000.0,
        "features": {"a": 1.0, "b": 2.0},
        "model_types": ["ModelA"],
        "model_weights": [1.0],
    }
    body.update(overrides)
    return body


def test_predict_returns_prediction(client, monkeypatch):
    monkeypatch.setattr(
        "services.prediction_service.inference.load_model",
        lambda name: _make_model(0.02),
    )

    resp = client.post("/predict", json=_payload())

    assert resp.status_code == 200
    body = resp.json()
    assert body["market"] == "jp"
    assert body["symbol"] == "7203"
    assert body["model_count"] == 1
    assert body["avg_pred_price"] == pytest.approx(1020.0)


def test_predict_all_models_missing_returns_200_with_zero_count(client, monkeypatch):
    """全モデル失敗でも 200 を返し model_count=0 で表現すること。"""
    monkeypatch.setattr("services.prediction_service.inference.load_model", lambda name: None)

    resp = client.post("/predict", json=_payload())

    assert resp.status_code == 200
    assert resp.json()["model_count"] == 0


def test_predict_weight_length_mismatch_returns_422(client):
    resp = client.post("/predict", json=_payload(model_types=["A", "B"], model_weights=[1.0]))
    assert resp.status_code == 422


def test_predict_missing_required_field_returns_422(client):
    body = _payload()
    del body["current_price"]
    resp = client.post("/predict", json=body)
    assert resp.status_code == 422


def test_health_ok_when_model_loads(client, monkeypatch):
    monkeypatch.setattr("services.prediction_service.app.load_model", lambda name: _make_model(0.0))

    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "UnifiedStockXGBoost" in body["loaded_models"]


def test_health_degraded_when_no_model_loads(client, monkeypatch):
    monkeypatch.setattr("services.prediction_service.app.load_model", lambda name: None)

    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["loaded_models"] == []
