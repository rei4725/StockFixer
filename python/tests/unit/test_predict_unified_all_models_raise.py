"""
predict_with_unified_model で全モデルが例外のときに例外を投げ直すことのテスト（#1140）
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.prediction.types import PredictionResult

_MOD = "src.prediction.predict_unified"


def _features() -> pd.DataFrame:
    return pd.DataFrame({"feat_a": [1.0, 2.0], "feat_b": [3.0, 4.0], "y": [0.0, 0.0]})


def _model(pred=None, error: Exception | None = None) -> MagicMock:
    model = MagicMock()
    model.feature_names_in_ = ["feat_a", "feat_b"]
    if error is not None:
        model.predict.side_effect = error
    else:
        model.predict.return_value = [pred]
    return model


def _patch_inputs(models: dict):
    """特徴量・現在価格・モデルを差し替える patch の並びを返す"""
    return [
        patch(f"{_MOD}.load_feature_data", return_value=_features()),
        patch(f"{_MOD}._fetch_current_price", return_value=1000.0),
        patch(f"{_MOD}.get_service_url", return_value=None),
        patch(f"{_MOD}.get_cached_model", side_effect=lambda name: models.get(name)),
        patch(
            f"{_MOD}.load_model_weights",
            side_effect=lambda market, symbol, names: [1.0 / len(names)] * len(names),
        ),
    ]


class _PatchedCase(unittest.TestCase):
    def _start(self, models: dict) -> None:
        for p in _patch_inputs(models):
            p.start()
            self.addCleanup(p.stop)


class TestPredictWithUnifiedModelAllRaise(_PatchedCase):
    def test_all_models_raise_reraises_with_last_cause(self):
        """全モデルの predict が例外なら RuntimeError を投げ、最後の例外が原因に付く"""
        from src.prediction.predict_unified import predict_with_unified_model

        self._start(
            {
                "UnifiedStockXGBoost": _model(error=TypeError("xgb")),
                "UnifiedStockLightGBM": _model(error=ValueError("lgbm")),
            }
        )

        with self.assertRaises(RuntimeError) as ctx:
            predict_with_unified_model("jp", "7203")
        self.assertIsInstance(ctx.exception.__cause__, ValueError)

    def test_one_model_raises_returns_rest(self):
        """1モデルだけ落ちると残りのモデルで予想を返し、model_count が減る"""
        from src.prediction.predict_unified import predict_with_unified_model

        self._start(
            {
                "UnifiedStockXGBoost": _model(error=TypeError("xgb")),
                "UnifiedStockLightGBM": _model(pred=0.01),
            }
        )

        result = predict_with_unified_model("jp", "7203")
        self.assertIsNotNone(result)
        self.assertEqual(result.model_count, 1)

    def test_no_model_loaded_returns_none(self):
        """どのモデルも読めていないときは None を返す"""
        from src.prediction.predict_unified import predict_with_unified_model

        self._start({})

        self.assertIsNone(predict_with_unified_model("jp", "7203"))


class TestPredictAllUnifiedOnError(_PatchedCase):
    @patch("src.prediction.prediction_pipeline.log_unified_prediction_summary")
    @patch("src.prediction.prediction_pipeline.get_active_symbols")
    @patch(f"{_MOD}.preload_models")
    def test_on_error_called_when_all_models_raise(self, _preload, mock_symbols, _summary):
        """全モデルが例外の銘柄で predict_all_unified の on_error が呼ばれる"""
        from src.prediction.prediction_pipeline import predict_all_unified

        mock_symbols.return_value = [("jp", "7203")]
        self._start(
            {
                "UnifiedStockXGBoost": _model(error=TypeError("xgb")),
                "UnifiedStockLightGBM": _model(error=ValueError("lgbm")),
            }
        )
        on_error = MagicMock()

        rows = predict_all_unified(max_workers=1, on_error=on_error)

        self.assertEqual(rows, [])
        on_error.assert_called_once_with("jp", "7203")


class TestMultiHorizonOneHorizonRaises(unittest.TestCase):
    @patch("src.prediction.multi_horizon.get_active_symbols")
    @patch(f"{_MOD}.preload_models")
    @patch(f"{_MOD}.predict_with_unified_model")
    def test_failed_horizon_is_left_empty(self, mock_predict, _preload, mock_symbols):
        """3d だけ全モデルが例外でも銘柄の行は残り、3d だけが欠ける"""
        from src.prediction.prediction_pipeline import predict_all_unified_multi_horizon

        mock_symbols.return_value = [("jp", "7203")]

        def _predict(market, symbol, model_types=None, horizon=1):
            if horizon == 3:
                raise RuntimeError("全モデルの予測が例外で失敗")
            return PredictionResult(
                market=market,
                symbol=symbol,
                current_price=1000.0,
                avg_pred_price=1010.0,
                diff_ratio=0.01,
                model_count=2,
            )

        mock_predict.side_effect = _predict

        rows = predict_all_unified_multi_horizon(horizons=[1, 3, 5])

        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0].diff_ratio_3d)
        self.assertEqual(rows[0].diff_ratio_5d, 0.01)


if __name__ == "__main__":
    unittest.main()
