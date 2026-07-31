"""推論ロジック（inference.py）のテスト。

実モデルファイルは使わず、feature_names_in_ を持つ MagicMock を注入する。
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from services.prediction_service.inference import clear_model_cache, run_inference
from services.prediction_service.types import PredictRequest


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_model_cache()
    yield
    clear_model_cache()


def _make_model(pred_return: float, expected_features: list[str] | None = None):
    """指定した変化率を返すモックモデルを作る。"""
    model = MagicMock()
    inner = MagicMock()
    if expected_features is not None:
        inner.feature_names_in_ = expected_features
    else:
        del inner.feature_names_in_
    model.model = inner
    model.predict.return_value = pd.Series([pred_return])
    return model


def _request(**overrides) -> PredictRequest:
    kwargs = {
        "market": "jp",
        "symbol": "7203",
        "current_price": 1000.0,
        "features": {"a": 1.0, "b": 2.0},
        "model_types": ["ModelA"],
        "model_weights": [1.0],
    }
    kwargs.update(overrides)
    return PredictRequest(**kwargs)


def test_single_model_prediction(monkeypatch):
    """1モデル・変化率0.02 → 予測価格1020, diff_ratio 0.02。"""
    model = _make_model(0.02, expected_features=["a", "b"])
    monkeypatch.setattr("services.prediction_service.inference.load_model", lambda name: model)

    resp = run_inference(_request())

    assert resp.model_count == 1
    assert resp.used_models == ["ModelA"]
    assert resp.avg_pred_price == pytest.approx(1020.0)
    assert resp.diff_ratio == pytest.approx(0.02)


def test_weighted_ensemble(monkeypatch):
    """2モデル（+0.02 / +0.04）を重み 0.5/0.5 → 平均 +0.03 相当。"""
    models = {
        "ModelA": _make_model(0.02, expected_features=["a", "b"]),
        "ModelB": _make_model(0.04, expected_features=["a", "b"]),
    }
    monkeypatch.setattr(
        "services.prediction_service.inference.load_model", lambda name: models[name]
    )

    resp = run_inference(_request(model_types=["ModelA", "ModelB"], model_weights=[0.5, 0.5]))

    assert resp.model_count == 2
    assert resp.avg_pred_price == pytest.approx(1030.0)
    assert resp.diff_ratio == pytest.approx(0.03)


def test_missing_features_filled_with_zero(monkeypatch):
    """モデルが期待する特徴量が不足していれば 0 で埋めて渡すこと。"""
    model = _make_model(0.01, expected_features=["a", "b", "c"])
    monkeypatch.setattr("services.prediction_service.inference.load_model", lambda name: model)

    run_inference(_request(features={"a": 1.0, "b": 2.0}))

    passed_df = model.predict.call_args[0][0]
    assert list(passed_df.columns) == ["a", "b", "c"]
    assert passed_df["c"].iloc[0] == 0


def test_all_models_fail_returns_zero_count(monkeypatch):
    """全モデルがロードできない場合 model_count=0 を返す（例外にしない）。"""
    monkeypatch.setattr("services.prediction_service.inference.load_model", lambda name: None)

    resp = run_inference(_request())

    assert resp.model_count == 0
    assert resp.used_models == []
    assert resp.avg_pred_price == 0.0
    assert resp.diff_ratio == 0.0


def test_partial_failure_renormalizes_weights(monkeypatch):
    """一部モデルが失敗した場合、成功分だけで重みを再正規化すること。"""
    model_b = _make_model(0.04, expected_features=["a", "b"])

    def _loader(name):
        return None if name == "ModelA" else model_b

    monkeypatch.setattr("services.prediction_service.inference.load_model", _loader)

    resp = run_inference(_request(model_types=["ModelA", "ModelB"], model_weights=[0.7, 0.3]))

    # ModelB のみ成功 → 重みは 1.0 に再正規化され、+0.04 がそのまま反映される
    assert resp.model_count == 1
    assert resp.used_models == ["ModelB"]
    assert resp.avg_pred_price == pytest.approx(1040.0)


def test_model_raising_exception_is_skipped(monkeypatch):
    """推論中に例外を投げるモデルはスキップして継続すること。"""
    bad = _make_model(0.0, expected_features=["a", "b"])
    bad.predict.side_effect = RuntimeError("boom")
    good = _make_model(0.02, expected_features=["a", "b"])

    def _loader(name):
        return bad if name == "ModelA" else good

    monkeypatch.setattr("services.prediction_service.inference.load_model", _loader)

    resp = run_inference(_request(model_types=["ModelA", "ModelB"], model_weights=[0.5, 0.5]))

    assert resp.model_count == 1
    assert resp.used_models == ["ModelB"]
    assert resp.avg_pred_price == pytest.approx(1020.0)
