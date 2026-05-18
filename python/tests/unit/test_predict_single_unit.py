"""ユニットテスト: 銘柄別予測処理（predict_single.py）"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


def _make_df(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.linspace(1000, 1100, n)
    return pd.DataFrame(
        {
            "Open": close - 5,
            "High": close + 10,
            "Low": close - 10,
            "Close": close,
            "Volume": np.full(n, 500_000, dtype=float),
        },
        index=idx,
    )


class TestResolveModelTypes(unittest.TestCase):
    def test_returns_given_list_when_provided(self):
        from src.prediction.predict_single import _resolve_model_types

        result = _resolve_model_types(1, ["CustomModel.joblib"])
        self.assertEqual(result, ["CustomModel.joblib"])

    def test_horizon_1_returns_default_models(self):
        from src.prediction.predict_single import _resolve_model_types

        result = _resolve_model_types(1)
        self.assertIn("StockXGBoostModel.joblib", result)
        self.assertIn("StockLightGBMModel.joblib", result)

    def test_horizon_3_returns_3d_models(self):
        from src.prediction.predict_single import _resolve_model_types

        result = _resolve_model_types(3)
        self.assertTrue(all("3d" in m for m in result))

    def test_horizon_5_returns_5d_models(self):
        from src.prediction.predict_single import _resolve_model_types

        result = _resolve_model_types(5)
        self.assertTrue(all("5d" in m for m in result))


class TestFetchCurrentPrice(unittest.TestCase):
    def test_returns_price_from_yfinance(self):
        from src.prediction.predict_single import _fetch_current_price

        df = _make_df()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [1050.0]}, index=pd.date_range("2024-01-31", periods=1)
        )

        with (
            patch("src.prediction.predict_single.get_ticker", return_value="7203.T"),
            patch("src.prediction.predict_single.yf.Ticker", return_value=mock_ticker),
        ):
            result = _fetch_current_price("jp", "7203", df)

        self.assertAlmostEqual(result, 1050.0, places=2)

    def test_fallback_to_df_close_when_yf_empty(self):
        from src.prediction.predict_single import _fetch_current_price

        df = _make_df()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        with (
            patch("src.prediction.predict_single.get_ticker", return_value="7203.T"),
            patch("src.prediction.predict_single.yf.Ticker", return_value=mock_ticker),
        ):
            result = _fetch_current_price("jp", "7203", df)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, df["Close"].iloc[-1], places=2)

    def test_fallback_when_yf_raises(self):
        from src.prediction.predict_single import _fetch_current_price

        df = _make_df()

        with (
            patch("src.prediction.predict_single.get_ticker", return_value="7203.T"),
            patch(
                "src.prediction.predict_single.yf.Ticker",
                side_effect=RuntimeError("network error"),
            ),
        ):
            result = _fetch_current_price("jp", "7203", df)

        self.assertIsNotNone(result)


class TestExplainPredictionShap(unittest.TestCase):
    def test_returns_none_when_shap_not_installed(self):
        import builtins

        from src.prediction.predict_single import explain_prediction_shap

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "shap":
                raise ImportError("no shap")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = explain_prediction_shap("jp", "7203")
        self.assertIsNone(result)

    def test_returns_none_when_model_not_found(self):
        from src.prediction.predict_single import explain_prediction_shap

        mock_shap = MagicMock()

        with (
            patch("src.prediction.predict_single.get_models_subdir", return_value="/nonexistent"),
            patch.dict("sys.modules", {"shap": mock_shap}),
        ):
            result = explain_prediction_shap("jp", "7203")
        self.assertIsNone(result)


class TestPredictSingleStock(unittest.TestCase):
    def test_returns_none_when_no_data(self):
        from src.prediction.predict_single import predict_single_stock

        with (
            patch(
                "src.prediction.predict_single.data_loader.get_stock_data",
                return_value=pd.DataFrame(),
            ),
        ):
            result = predict_single_stock("jp", "7203")
        self.assertIsNone(result)

    def test_returns_none_when_no_model_files(self):
        from src.prediction.predict_single import predict_single_stock

        df = _make_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("src.prediction.predict_single.data_loader.get_stock_data", return_value=df),
                patch("src.prediction.predict_single.add_technical_indicators", return_value=df),
                patch(
                    "src.prediction.predict_single.create_basic_lag_features",
                    return_value=(df, pd.Series()),
                ),
                patch("src.prediction.predict_single.get_models_subdir", return_value=tmpdir),
                patch("src.prediction.predict_single.get_ticker", return_value="7203.T"),
                patch(
                    "src.prediction.predict_single.yf.Ticker",
                    return_value=MagicMock(history=MagicMock(return_value=pd.DataFrame())),
                ),
            ):
                result = predict_single_stock("jp", "7203")
        self.assertIsNone(result)


class TestPredictMultiHorizon(unittest.TestCase):
    def test_returns_none_when_all_horizons_fail(self):
        from src.prediction.predict_single import predict_single_stock_multi_horizon

        with patch("src.prediction.predict_single.predict_single_stock", return_value=None):
            result = predict_single_stock_multi_horizon("jp", "7203")
        self.assertIsNone(result)

    def test_returns_result_when_at_least_one_horizon_succeeds(self):
        from src.prediction.predict_single import predict_single_stock_multi_horizon
        from src.prediction.types import PredictionResult

        mock_result = PredictionResult(
            market="jp",
            symbol="7203",
            current_price=1000.0,
            avg_pred_price=1020.0,
            diff_ratio=0.02,
            model_count=2,
        )

        def fake_predict(market, symbol, horizon=1):
            if horizon == 1:
                return mock_result
            return None

        with patch("src.prediction.predict_single.predict_single_stock", side_effect=fake_predict):
            result = predict_single_stock_multi_horizon("jp", "7203", horizons=[1, 3])
        self.assertIsNotNone(result)
        self.assertEqual(result.market, "jp")

    def test_custom_horizons_are_used(self):
        from src.prediction.predict_single import predict_single_stock_multi_horizon
        from src.prediction.types import PredictionResult

        called_horizons = []

        def fake_predict(market, symbol, horizon=1):
            called_horizons.append(horizon)
            return PredictionResult(
                market=market,
                symbol=symbol,
                current_price=1000.0,
                avg_pred_price=1010.0,
                diff_ratio=0.01,
                model_count=1,
            )

        with patch("src.prediction.predict_single.predict_single_stock", side_effect=fake_predict):
            predict_single_stock_multi_horizon("jp", "7203", horizons=[1, 5])

        self.assertIn(1, called_horizons)
        self.assertIn(5, called_horizons)


class TestFetchCurrentPriceDataFrameClose(unittest.TestCase):
    def test_close_as_dataframe_uses_first_column(self):
        import pandas as pd

        from src.prediction.predict_single import _fetch_current_price

        df = _make_df()
        # Close列をDataFrameにする（MultiIndex yfinance スタイル）
        df_multi = df.copy()
        df_multi["Close"] = pd.DataFrame({"A": df["Close"].values}, index=df.index)

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()  # empty → fallback

        with (
            patch("src.prediction.predict_single.get_ticker", return_value="7203.T"),
            patch("src.prediction.predict_single.yf.Ticker", return_value=mock_ticker),
        ):
            result = _fetch_current_price("jp", "7203", df_multi)

        self.assertIsNotNone(result)


class TestExplainPredictionShapBody(unittest.TestCase):
    def test_returns_result_with_mocked_shap(self):
        import tempfile

        import numpy as np

        from src.prediction.predict_single import explain_prediction_shap

        df = _make_df()

        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_shap_values = MagicMock()
        mock_shap_values.__class__ = type("ShapValues", (), {})
        mock_shap.TreeExplainer.return_value = mock_explainer
        # shap_values は numpy array を返す
        n_features = 5
        mock_explainer.shap_values.return_value = np.array([[0.1, -0.2, 0.3, -0.1, 0.05]])

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.02]

        mock_mm = MagicMock()
        mock_mm.load_model.return_value = mock_model

        with tempfile.TemporaryDirectory() as tmpdir:
            # モデルファイルを作成
            model_path = os.path.join(tmpdir, "StockXGBoostModel.joblib")
            with open(model_path, "wb") as f:
                f.write(b"dummy")

            X_df = pd.DataFrame(
                np.random.rand(10, n_features), columns=[f"feature_{i}" for i in range(n_features)]
            )

            with (
                patch.dict("sys.modules", {"shap": mock_shap}),
                patch("src.prediction.predict_single.get_models_subdir", return_value=tmpdir),
                patch("src.prediction.predict_single.data_loader.get_stock_data", return_value=df),
                patch("src.prediction.predict_single.add_technical_indicators", return_value=df),
                patch(
                    "src.prediction.predict_single.create_basic_lag_features",
                    return_value=(X_df, pd.Series()),
                ),
                patch("src.prediction.predict_single.ModelManager", return_value=mock_mm),
                patch("src.prediction.predict_single.normalize_col", side_effect=lambda x: x),
            ):
                result = explain_prediction_shap("jp", "7203")

        self.assertIsNotNone(result)
        self.assertIn("symbol", result)
        self.assertIn("top_features", result)

    def test_returns_none_when_data_empty(self):
        import tempfile

        from src.prediction.predict_single import explain_prediction_shap

        mock_shap = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "StockXGBoostModel.joblib")
            with open(model_path, "wb") as f:
                f.write(b"dummy")

            with (
                patch.dict("sys.modules", {"shap": mock_shap}),
                patch("src.prediction.predict_single.get_models_subdir", return_value=tmpdir),
                patch(
                    "src.prediction.predict_single.data_loader.get_stock_data",
                    return_value=pd.DataFrame(),
                ),
            ):
                result = explain_prediction_shap("jp", "7203")

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
