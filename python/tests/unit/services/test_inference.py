"""推論ロジック（inference.py）のテスト。

実モデルファイルは使わず、feature_names_in_ を持つ MagicMock を注入する。
"""

from unittest.mock import MagicMock

import numpy as np
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
    """指定した変化率を返すモックモデルを作る。

    load_model() が返すのは joblib.load() の戻り値そのもの（生の sklearn /
    xgboost 推定器）なので、モックも feature_names_in_ を直下に持たせる。
    """
    model = MagicMock()
    if expected_features is not None:
        model.feature_names_in_ = expected_features
    else:
        del model.feature_names_in_
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


def test_real_load_model_path_aligns_features(tmp_path, monkeypatch):
    """load_model() を monkeypatch せず、実際に joblib で保存した推定器を読んで
    特徴量アラインメントが働くことを検証する。

    他のテストは load_model() を差し替えているため、joblib.load() の戻り値の
    形（生の推定器か、ラッパーか）の食い違いを検出できない。ここだけは実際の
    ロード経路を通す。
    """
    import joblib
    from sklearn.linear_model import LinearRegression

    # feature_names_in_ を持つ実推定器を作って保存する
    estimator = LinearRegression()
    train_X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
    estimator.fit(train_X, pd.Series([0.0, 0.0]))
    joblib.dump(estimator, tmp_path / "RealModel.joblib")

    monkeypatch.setattr("services.prediction_service.inference.MODEL_DIR", str(tmp_path))

    # 特徴量 "c" を欠いた状態で投げても、0 埋めされて推論が成功すること
    resp = run_inference(_request(features={"a": 1.0, "b": 2.0}, model_types=["RealModel"]))

    assert resp.model_count == 1
    assert resp.used_models == ["RealModel"]


def _fit_legacy_lightgbm(feature_names: list[str]):
    """本番の UnifiedStockLightGBM と同じ属性の形をした実推定器を返す（#615）。

    lightgbm が ``feature_names_in_`` を持つようになったのは 4.5.0 からで、
    本番モデルはそれ以前に pickle されているためこの属性を持たない。裏にある
    private フラグを消すことで、その状態を実物の推定器のまま再現する。
    """
    import lightgbm as lgb

    rng = np.random.default_rng(42)
    train_X = pd.DataFrame({name: rng.normal(size=60) for name in feature_names})
    train_y = pd.Series(rng.normal(size=60))
    estimator = lgb.LGBMRegressor(
        n_estimators=5, min_child_samples=5, num_leaves=3, verbose=-1
    ).fit(train_X, train_y)
    del estimator.__dict__["_fitted_with_feature_names"]
    return estimator


def test_lightgbm_without_feature_names_in_is_still_aligned(monkeypatch):
    """feature_names_in_ を持たない LightGBM でもアラインメントされること（#615）。

    MagicMock ではなく実物の LGBMRegressor を使う。MagicMock の predict() は
    列数が食い違う DataFrame も受け付けてしまい、このバグを検出できない。
    """
    expected = ["Close_lag1", "feature_a", "feature_b"]
    model = _fit_legacy_lightgbm(expected)
    assert not hasattr(model, "feature_names_in_")

    monkeypatch.setattr("services.prediction_service.inference.load_model", lambda name: model)

    # 期待特徴量 feature_b を欠き、未知の余剰特徴量を2つ含む状態で投げる
    resp = run_inference(
        _request(
            features={
                "Close_lag1": 1000.0,
                "feature_a": 0.5,
                "unused_extra": 9.9,
                "another_extra": 1.0,
            },
            model_types=["UnifiedStockLightGBM"],
            model_weights=[1.0],
        )
    )

    assert resp.model_count == 1
    assert resp.used_models == ["UnifiedStockLightGBM"]


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
