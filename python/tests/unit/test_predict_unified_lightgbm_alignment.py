"""回帰テスト: 統合モデルの特徴量アラインメントが LightGBM でも効くこと（#615）。

本番の ``UnifiedStockLightGBM.joblib`` は ``feature_names_in_`` が追加された
lightgbm 4.5.0 より前のバージョンで pickle されているため、ロードしても
``feature_names_in_`` を持たず、LightGBM 独自の ``feature_name_`` にしか
特徴量名を持たない。``predict_with_unified_model`` はアラインメントの可否を
``feature_names_in_`` の有無だけで判定していたため、LightGBM ではアラインメントが
丸ごとスキップされ、``date`` 列を含む生の DataFrame が predict() に渡って
毎回 DTypePromotionError で失敗していた（実質 XGBoost 単独で稼働）。

MagicMock ではなく実物の LGBMRegressor を使うのは、MagicMock の predict() が
任意の DataFrame を受け付けてしまい、バグのあるコードでも通ってしまうため
（このバグが見逃された原因そのもの）。
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# 実推定器を学習させる際のサンプル数。LightGBM が分割を作れる程度に確保する。
_N_SAMPLES = 60
_EXPECTED_FEATURES = ["Close_lag1", "feature_a", "feature_b"]

# lightgbm の feature_names_in_ プロパティの裏にある private フラグ。
# 4.5.0 より前の pickle にはこの属性自体が存在せず、プロパティは AttributeError を
# 送出する。本番モデルのその状態を再現するために削除する。
_LGBM_FEATURE_NAMES_FLAG = "_fitted_with_feature_names"


def _training_frame(feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame({name: rng.normal(size=_N_SAMPLES) for name in feature_names})
    y = pd.Series(X[feature_names[0]] * 0.01 + rng.normal(scale=0.001, size=_N_SAMPLES))
    return X, y


def _fit_legacy_lightgbm_wrapper(feature_names: list[str]):
    """本番と同じ状態（feature_names_in_ 無し / feature_name_ 有り）の LightGBM を返す。

    ラッパーは本番同様 ``.model`` に推定器を持つ（get_cached_model の戻り値と同型）。
    """
    from src.prediction.models.lightgbm import LightGBMModel

    X, y = _training_frame(feature_names)
    model = LightGBMModel(
        model_name="UnifiedStockLightGBM", n_estimators=5, min_child_samples=5, num_leaves=3
    )
    model.train(X, y)
    # 4.5.0 未満で pickle された本番モデルの状態を再現する
    del model.model.__dict__[_LGBM_FEATURE_NAMES_FLAG]
    return model


def _fit_xgboost_wrapper(feature_names: list[str]):
    from src.prediction.models.xgboost import XGBoostModel

    X, y = _training_frame(feature_names)
    model = XGBoostModel(model_name="UnifiedStockXGBoost", n_estimators=5, max_depth=2)
    model.train(X, y)
    return model


def _feature_frame_with_date() -> pd.DataFrame:
    """本番の load_stock_features 相当。date 列と未知の余剰列を含む。"""
    n = 10
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "y": np.linspace(0.001, 0.01, n),
            "Close_lag1": np.linspace(998, 1098, n),
            "feature_a": np.linspace(-1, 1, n),
            "feature_b": np.linspace(1, -1, n),
            "unused_extra_feature": np.linspace(0, 5, n),
        }
    )


class TestLegacyLightGBMPrecondition(unittest.TestCase):
    """テストが前提としている「本番モデルの属性の形」自体を固定する。"""

    def test_legacy_model_exposes_only_feature_name(self):
        model = _fit_legacy_lightgbm_wrapper(_EXPECTED_FEATURES)

        self.assertFalse(hasattr(model.model, "feature_names_in_"))
        self.assertEqual(list(model.model.feature_name_), _EXPECTED_FEATURES)

    def test_xgboost_exposes_feature_names_in(self):
        model = _fit_xgboost_wrapper(_EXPECTED_FEATURES)

        self.assertEqual(list(model.model.feature_names_in_), _EXPECTED_FEATURES)


def _run_predict(models_by_name: dict, model_types: list[str], weights: list[float]):
    """本番相当の入力で predict_with_unified_model を1銘柄ぶん走らせる。"""
    from src.prediction.predict_unified import predict_with_unified_model

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame(
        {"Close": [1050.0]}, index=pd.date_range("2024-01-31", periods=1)
    )

    with (
        patch(
            "src.prediction.predict_unified.load_feature_data",
            return_value=_feature_frame_with_date(),
        ),
        patch(
            "src.prediction.predict_unified.get_cached_model",
            side_effect=lambda name: models_by_name.get(name),
        ),
        patch("src.prediction.predict_unified.get_ticker", return_value="7203.T"),
        patch("src.prediction.predict_unified.load_model_weights", return_value=weights),
        patch("src.prediction.predict_unified.get_service_url", return_value=None),
        patch("src.prediction.predict_unified.yf.Ticker", return_value=mock_ticker),
    ):
        return predict_with_unified_model("jp", "7203", model_types=model_types)


class TestPredictWithUnifiedModelEnsemble(unittest.TestCase):
    def setUp(self):
        from src.prediction import predict_unified

        predict_unified._model_cache.clear()

    def _run(self, models_by_name: dict, model_types: list[str], weights: list[float]):
        return _run_predict(models_by_name, model_types, weights)

    def test_lightgbm_alone_produces_a_prediction(self):
        """修正前は date 列混入で DTypePromotionError → None が返っていた。"""
        models = {"UnifiedStockLightGBM": _fit_legacy_lightgbm_wrapper(_EXPECTED_FEATURES)}

        result = self._run(models, ["UnifiedStockLightGBM"], [1.0])

        self.assertIsNotNone(result)
        self.assertEqual(result.model_count, 1)

    def test_both_models_participate_in_the_ensemble(self):
        """本来の2モデルアンサンブルが成立すること（本番は model_count=1 だった）。"""
        models = {
            "UnifiedStockXGBoost": _fit_xgboost_wrapper(_EXPECTED_FEATURES),
            "UnifiedStockLightGBM": _fit_legacy_lightgbm_wrapper(_EXPECTED_FEATURES),
        }

        result = self._run(models, ["UnifiedStockXGBoost", "UnifiedStockLightGBM"], [0.5, 0.5])

        self.assertIsNotNone(result)
        self.assertEqual(result.model_count, 2)

    def test_models_with_differing_feature_sets_both_succeed(self):
        """本番同様、2モデルの期待特徴量数が食い違っていても各々に揃うこと。"""
        models = {
            "UnifiedStockXGBoost": _fit_xgboost_wrapper(_EXPECTED_FEATURES),
            "UnifiedStockLightGBM": _fit_legacy_lightgbm_wrapper(_EXPECTED_FEATURES[:2]),
        }

        result = self._run(models, ["UnifiedStockXGBoost", "UnifiedStockLightGBM"], [0.5, 0.5])

        self.assertIsNotNone(result)
        self.assertEqual(result.model_count, 2)


class _RecordingModel:
    """特徴量名を解決できないモデル（アラインメントがスキップされる経路）。"""

    def __init__(self) -> None:
        self.received: pd.DataFrame | None = None

    def predict(self, X: pd.DataFrame) -> pd.Series:
        self.received = X
        return pd.Series([0.01], index=X.index)


class TestFeatureColumnFiltering(unittest.TestCase):
    """アラインメントを解決できない場合でも日付列は predict に渡らないこと。"""

    def test_date_column_never_reaches_predict(self):
        recorder = _RecordingModel()

        result = _run_predict({"UnifiedStockXGBoost": recorder}, ["UnifiedStockXGBoost"], [1.0])

        self.assertIsNotNone(result)
        self.assertIsNotNone(recorder.received)
        self.assertNotIn("date", recorder.received.columns)
        self.assertFalse(
            any(
                pd.api.types.is_datetime64_any_dtype(recorder.received[c])
                for c in recorder.received.columns
            )
        )


if __name__ == "__main__":
    unittest.main()
