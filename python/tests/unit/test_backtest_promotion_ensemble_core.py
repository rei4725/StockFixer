"""
統合モデルのアンサンブルの芯（ensemble_diff_ratio）のテスト

バックテストが本番と同じ芯を、候補のモデルの実物と本番の重みの名前で呼べること、
本番の関数と同じ値を出すことを確かめる。
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.prediction import predict_unified
from src.prediction.predict_unified import ensemble_diff_ratio, predict_with_unified_model
from src.prediction.types import EnsembleOutcome


def _features() -> pd.DataFrame:
    # atr_lag1=10・価格100 なら上限は ±(10*倍率/100) で 0.04 より広く、クリップしない
    return pd.DataFrame(
        {
            "feat_a": [1.0, 2.0],
            "atr_lag1": [10.0, 10.0],
            "market_encoded": [0, 0],
            "y": [0.0, 0.0],
        }
    )


def _row() -> pd.DataFrame:
    return _features().drop(columns=["y"]).iloc[[-1]].copy()


def _model(pred: float, features=("feat_a", "atr_lag1", "market_encoded")) -> MagicMock:
    # spec で属性を絞る。素の MagicMock は .model を生やし、期待特徴量を解決できなくなる
    model = MagicMock(spec=["predict", "feature_names_in_"])
    model.feature_names_in_ = list(features)
    model.predict.return_value = [pred]
    return model


def _patch_live(monkeypatch, models: dict, weights: list[float]) -> dict:
    """本番の関数の入力を差し替え、記録の呼び出しを集める"""
    calls: dict = {"zero_fill": [], "clip": []}
    monkeypatch.setattr(predict_unified, "load_feature_data", lambda m, s: _features())
    monkeypatch.setattr(predict_unified, "_fetch_current_price", lambda m, s, d: 100.0)
    monkeypatch.setattr(predict_unified, "get_service_url", lambda: None)
    monkeypatch.setattr(predict_unified, "get_cached_model", lambda name: models.get(name))
    monkeypatch.setattr(predict_unified, "load_model_weights", lambda m, s, names: list(weights))
    monkeypatch.setattr(
        predict_unified,
        "record_zero_fill",
        lambda *a: calls["zero_fill"].append(a),
    )
    monkeypatch.setattr(predict_unified, "record_clip", lambda *a: calls["clip"].append(a))
    return calls


def test_ensemble_core_matches_live_function(monkeypatch):
    m1, m2 = _model(0.02), _model(0.04)
    out = ensemble_diff_ratio(
        _row(), 100.0, "us", "A", 1, [("X", m1), ("L", m2)], weights_fn=lambda *a: [0.25, 0.75]
    )
    assert isinstance(out, EnsembleOutcome)
    assert out.diff_ratio == pytest.approx(0.035)
    assert out.diff_ratio_raw == pytest.approx(0.035)
    assert out.clip_max_ratio is not None and out.clip_max_ratio > 0.035
    assert out.model_count == 2
    assert out.zero_filled == []

    _patch_live(
        monkeypatch,
        {"UnifiedStockXGBoost": _model(0.02), "UnifiedStockLightGBM": _model(0.04)},
        [0.25, 0.75],
    )
    live = predict_with_unified_model("us", "A")
    assert live.diff_ratio == pytest.approx(out.diff_ratio, abs=1e-12)
    assert live.model_count == out.model_count


def test_ensemble_core_clips_like_live(monkeypatch):
    """クリップに当たる値でも芯と本番が同じ値になり、クリップ前の値も残る"""
    out = ensemble_diff_ratio(
        _row(), 100.0, "us", "A", 1, [("X", _model(5.0))], weights_fn=lambda *a: [1.0]
    )
    assert out.diff_ratio_raw == pytest.approx(5.0)
    assert out.diff_ratio == pytest.approx(out.clip_max_ratio)

    calls = _patch_live(monkeypatch, {"UnifiedStockXGBoost": _model(5.0)}, [1.0])
    live = predict_with_unified_model("us", "A")
    assert live.diff_ratio == pytest.approx(out.diff_ratio, abs=1e-12)
    assert calls["clip"] == [("us", 1, pytest.approx(5.0), pytest.approx(out.clip_max_ratio))]


def test_ensemble_core_does_not_record_run_stats(monkeypatch):
    zero_fill = MagicMock()
    clip = MagicMock()
    monkeypatch.setattr(predict_unified, "record_zero_fill", zero_fill)
    monkeypatch.setattr(predict_unified, "record_clip", clip)

    # feat_missing が行に無いので0で埋まる
    model = _model(0.01, features=("feat_a", "feat_missing"))
    out = ensemble_diff_ratio(
        _row(), 100.0, "us", "A", 1, [("X", model)], weights_fn=lambda *a: [1.0]
    )

    assert out.zero_filled == ["feat_missing"]
    zero_fill.assert_not_called()
    clip.assert_not_called()


def test_live_function_still_records_run_stats(monkeypatch):
    calls = _patch_live(
        monkeypatch,
        {"UnifiedStockXGBoost": _model(0.01, features=("feat_a", "feat_missing"))},
        [1.0],
    )
    predict_with_unified_model("us", "A")
    assert calls["zero_fill"] == [("us", 1, ["feat_missing"])]
    assert len(calls["clip"]) == 1


def test_ensemble_core_uses_weight_names_not_model_objects():
    seen = []

    def weights_fn(market, symbol, names):
        seen.append((market, symbol, names))
        return [0.5, 0.5]

    ensemble_diff_ratio(
        _row(),
        100.0,
        "jp",
        "7203",
        1,
        [("UnifiedStockXGBoost", _model(0.01)), ("UnifiedStockLightGBM", _model(0.02))],
        weights_fn=weights_fn,
    )
    assert seen == [("jp", "7203", ["UnifiedStockXGBoost", "UnifiedStockLightGBM"])]


def test_ensemble_core_weights_only_succeeded_models():
    """落ちたモデルは重みの名前から外れる（本番と同じ）"""
    seen = []
    bad = MagicMock(spec=["predict", "feature_names_in_"])
    bad.feature_names_in_ = ["feat_a"]
    bad.predict.side_effect = ValueError("boom")

    out = ensemble_diff_ratio(
        _row(),
        100.0,
        "us",
        "A",
        1,
        [("X", bad), ("L", _model(0.02))],
        weights_fn=lambda m, s, names: seen.append(names) or [1.0],
    )
    assert seen == [["L"]]
    assert out.model_count == 1
    assert out.diff_ratio == pytest.approx(0.02)


def test_ensemble_core_all_models_raise():
    bad1 = MagicMock(spec=["predict", "feature_names_in_"])
    bad2 = MagicMock(spec=["predict", "feature_names_in_"])
    bad1.feature_names_in_ = bad2.feature_names_in_ = ["feat_a"]
    bad1.predict.side_effect = TypeError("x")
    bad2.predict.side_effect = ValueError("l")

    with pytest.raises(RuntimeError, match="market=us symbol=A") as exc:
        ensemble_diff_ratio(_row(), 100.0, "us", "A", 1, [("X", bad1), ("L", bad2)])
    assert isinstance(exc.value.__cause__, ValueError)


def test_ensemble_core_no_models_returns_none():
    assert ensemble_diff_ratio(_row(), 100.0, "us", "A", 1, []) is None
    assert ensemble_diff_ratio(_row(), 100.0, "us", "A", 1, [("X", None)]) is None


def test_live_load_error_counts_as_model_error(monkeypatch):
    """本番の関数でモデルの読み込みが例外なら、今までどおりそのモデルだけ飛ばす"""
    _patch_live(monkeypatch, {"UnifiedStockLightGBM": _model(0.02)}, [1.0])

    def load(name):
        if name == "UnifiedStockXGBoost":
            raise OSError("broken file")
        return _model(0.02)

    monkeypatch.setattr(predict_unified, "get_cached_model", load)
    live = predict_with_unified_model("us", "A")
    assert live.model_count == 1
    assert live.diff_ratio == pytest.approx(0.02)

    def load_fail(name):
        raise OSError(name)

    monkeypatch.setattr(predict_unified, "get_cached_model", load_fail)
    with pytest.raises(RuntimeError, match="market=us symbol=A") as exc:
        predict_with_unified_model("us", "A")
    assert str(exc.value.__cause__) == "UnifiedStockLightGBM"


def test_live_records_zero_fill_before_weights_and_on_all_raise(monkeypatch):
    """0埋めの記録は重みの読み込みより前、全モデルが例外でも出る（今までと同じ順）"""
    order = []
    calls = _patch_live(monkeypatch, {}, [1.0])
    monkeypatch.setattr(
        predict_unified, "record_zero_fill", lambda *a: order.append(("zero_fill", a))
    )
    monkeypatch.setattr(
        predict_unified, "load_model_weights", lambda m, s, n: order.append(("weights",)) or [1.0]
    )
    monkeypatch.setattr(predict_unified, "record_clip", lambda *a: order.append(("clip",)))
    model = _model(0.01, features=("feat_a", "feat_missing"))
    monkeypatch.setattr(
        predict_unified,
        "get_cached_model",
        lambda name: model if name == "UnifiedStockXGBoost" else None,
    )
    predict_with_unified_model("us", "A")
    assert order == [("zero_fill", ("us", 1, ["feat_missing"])), ("weights",), ("clip",)]
    assert calls["clip"] == []

    order.clear()
    bad = MagicMock(spec=["predict", "feature_names_in_"])
    bad.feature_names_in_ = ["feat_a", "feat_missing"]
    bad.predict.side_effect = ValueError("boom")
    monkeypatch.setattr(predict_unified, "get_cached_model", lambda name: bad)
    with pytest.raises(RuntimeError):
        predict_with_unified_model("us", "A")
    assert order == [("zero_fill", ("us", 1, ["feat_missing"]))]
