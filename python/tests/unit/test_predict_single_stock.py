"""predict_single_stock モジュールのユニットテスト"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.domain.types import PredictionResult

# ──────────────────────────────────────────────────────────────────
# ヘルパー
# ──────────────────────────────────────────────────────────────────


def _make_price_df(n=30):
    """テスト用の株価DataFrame"""
    idx = pd.date_range("2026-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": np.random.uniform(99, 101, n),
            "High": np.random.uniform(101, 103, n),
            "Low": np.random.uniform(97, 99, n),
            "Close": np.linspace(100, 110, n),
            "Volume": np.random.randint(1000, 5000, n).astype(float),
        },
        index=idx,
    )


# ──────────────────────────────────────────────────────────────────
# _resolve_model_types  （純粋関数 → モック不要）
# ──────────────────────────────────────────────────────────────────


class TestResolveModelTypes(unittest.TestCase):
    def _fn(self, horizon, model_types=None):
        from src.models.predict_single_stock import _resolve_model_types

        return _resolve_model_types(horizon, model_types)

    def test_horizon1_default(self):
        result = self._fn(1)
        self.assertEqual(result, ["StockXGBoostModel.joblib", "StockLightGBMModel.joblib"])

    def test_horizon3_suffix(self):
        result = self._fn(3)
        self.assertIn("StockXGBoostModel_3d.joblib", result)
        self.assertIn("StockLightGBMModel_3d.joblib", result)

    def test_horizon10_suffix(self):
        result = self._fn(10)
        self.assertIn("StockXGBoostModel_10d.joblib", result)

    def test_explicit_model_types_returned_as_is(self):
        custom = ["MyModel.joblib"]
        result = self._fn(1, model_types=custom)
        self.assertEqual(result, custom)

    def test_explicit_overrides_any_horizon(self):
        custom = ["X.joblib", "Y.joblib"]
        self.assertEqual(self._fn(5, model_types=custom), custom)

    def test_returns_list_type(self):
        self.assertIsInstance(self._fn(1), list)
        self.assertIsInstance(self._fn(3), list)


# ──────────────────────────────────────────────────────────────────
# _fetch_current_price  （yfinance をモック）
# ──────────────────────────────────────────────────────────────────


class TestFetchCurrentPrice(unittest.TestCase):
    def _fn(self, df, yf_hist=None, yf_raises=False):
        from src.models.predict_single_stock import _fetch_current_price

        mock_hist = pd.DataFrame({"Close": [123.45]}) if yf_hist is None else yf_hist
        mock_ticker = MagicMock()
        if yf_raises:
            mock_ticker.history.side_effect = RuntimeError("network error")
        else:
            mock_ticker.history.return_value = mock_hist

        with patch("src.models.predict_single_stock.yf.Ticker", return_value=mock_ticker), patch(
            "src.models.predict_single_stock.get_ticker", return_value="AAPL"
        ):
            return _fetch_current_price("us", "AAPL", df)

    def test_uses_yfinance_price_when_available(self):
        df = _make_price_df()
        result = self._fn(df)
        self.assertAlmostEqual(result, 123.45)

    def test_falls_back_to_df_when_yf_empty(self):
        df = _make_price_df()
        result = self._fn(df, yf_hist=pd.DataFrame())
        # dfの末尾Close値になること
        expected = float(df["Close"].iloc[-1])
        self.assertAlmostEqual(result, expected)

    def test_falls_back_to_df_when_yf_raises(self):
        df = _make_price_df()
        result = self._fn(df, yf_raises=True)
        expected = float(df["Close"].iloc[-1])
        self.assertAlmostEqual(result, expected)

    def test_returns_float(self):
        df = _make_price_df()
        result = self._fn(df)
        self.assertIsInstance(result, float)

    def test_returns_none_when_df_has_no_close_and_yf_fails(self):
        from src.models.predict_single_stock import _fetch_current_price

        df = pd.DataFrame({"Open": [100.0]})  # Closeなし

        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = RuntimeError("err")
        with patch("src.models.predict_single_stock.yf.Ticker", return_value=mock_ticker), patch(
            "src.models.predict_single_stock.get_ticker", return_value="AAPL"
        ):
            result = _fetch_current_price("us", "AAPL", df)
        self.assertIsNone(result)


# ──────────────────────────────────────────────────────────────────
# _build_prediction_result  （純粋関数）
# ──────────────────────────────────────────────────────────────────


class TestBuildPredictionResult(unittest.TestCase):
    def _fn(self, market, symbol, current, prices, returns):
        from src.models.predict_single_stock import _build_prediction_result

        return _build_prediction_result(market, symbol, current, prices, returns)

    def test_returns_none_on_empty_prices(self):
        result = self._fn("us", "AAPL", 100.0, [], [])
        self.assertIsNone(result)

    def test_returns_none_when_current_price_none(self):
        result = self._fn("us", "AAPL", None, [105.0], [0.05])
        self.assertIsNone(result)

    def test_avg_pred_price_single_model(self):
        result = self._fn("us", "AAPL", 100.0, [110.0], [0.10])
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.avg_pred_price, 110.0)

    def test_avg_pred_price_multiple_models(self):
        result = self._fn("us", "AAPL", 100.0, [110.0, 90.0], [0.10, -0.10])
        self.assertAlmostEqual(result.avg_pred_price, 100.0)

    def test_diff_ratio_positive(self):
        result = self._fn("us", "AAPL", 100.0, [110.0], [0.10])
        self.assertAlmostEqual(result.diff_ratio, 0.10)

    def test_diff_ratio_negative(self):
        result = self._fn("us", "AAPL", 100.0, [90.0], [-0.10])
        self.assertAlmostEqual(result.diff_ratio, -0.10)

    def test_model_count(self):
        result = self._fn("us", "AAPL", 100.0, [110.0, 108.0], [0.10, 0.08])
        self.assertEqual(result.model_count, 2)

    def test_confidence_ratio_single_model(self):
        """1モデルのとき std=0 → confidence_ratio=1.0"""
        result = self._fn("us", "AAPL", 100.0, [105.0], [0.05])
        self.assertAlmostEqual(result.confidence_ratio, 1.0)

    def test_confidence_ratio_decreases_with_higher_std(self):
        """std が大きいほど confidence_ratio が小さくなること"""
        result_low_std = self._fn("us", "AAPL", 100.0, [105.0, 104.0], [0.05, 0.04])
        result_high_std = self._fn("us", "AAPL", 100.0, [120.0, 80.0], [0.20, -0.20])
        self.assertGreater(result_low_std.confidence_ratio, result_high_std.confidence_ratio)

    def test_returns_prediction_result_type(self):
        result = self._fn("jp", "7203", 2000.0, [2100.0], [0.05])
        self.assertIsInstance(result, PredictionResult)

    def test_market_symbol_preserved(self):
        result = self._fn("jp", "7203", 2000.0, [2100.0], [0.05])
        self.assertEqual(result.market, "jp")
        self.assertEqual(result.symbol, "7203")


# ──────────────────────────────────────────────────────────────────
# _run_single_model_prediction  （feature/model をモック）
# ──────────────────────────────────────────────────────────────────


class TestRunSingleModelPrediction(unittest.TestCase):
    def _call(self, model_path, df, current_price=100.0, pred_return=0.05):
        from src.models.predict_single_stock import _run_single_model_prediction

        n = len(df)
        fake_X = pd.DataFrame({"f1": np.ones(n)}, index=df.index)
        fake_y = pd.Series(np.zeros(n), index=df.index)

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([pred_return])
        mock_mm = MagicMock()
        mock_mm.load_model.return_value = mock_model

        with patch(
            "src.models.predict_single_stock.add_technical_indicators", return_value=df
        ), patch(
            "src.models.predict_single_stock.create_basic_lag_features",
            return_value=(fake_X, fake_y),
        ), patch(
            "src.models.predict_single_stock.ModelManager", return_value=mock_mm
        ):
            return _run_single_model_prediction(model_path, "us", "AAPL", df, current_price)

    def test_returns_tuple_on_success(self):
        df = _make_price_df()
        result = self._call("some/path/StockXGBoostModel.joblib", df)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    def test_pred_price_correct(self):
        df = _make_price_df()
        result = self._call(
            "path/StockXGBoostModel.joblib", df, current_price=100.0, pred_return=0.10
        )
        pred_price, pred_return = result
        self.assertAlmostEqual(pred_price, 110.0)
        self.assertAlmostEqual(pred_return, 0.10)

    def test_returns_none_when_features_empty(self):
        from src.models.predict_single_stock import _run_single_model_prediction

        df = _make_price_df(5)
        empty_X = pd.DataFrame()
        empty_y = pd.Series(dtype=float)

        with patch(
            "src.models.predict_single_stock.add_technical_indicators", return_value=df
        ), patch(
            "src.models.predict_single_stock.create_basic_lag_features",
            return_value=(empty_X, empty_y),
        ), patch(
            "src.models.predict_single_stock.ModelManager"
        ):
            result = _run_single_model_prediction("path/model.joblib", "us", "AAPL", df, 100.0)
        self.assertIsNone(result)

    def test_handles_series_prediction(self):
        """model.predict が pd.Series を返すケース"""
        from src.models.predict_single_stock import _run_single_model_prediction

        df = _make_price_df()
        n = len(df)
        fake_X = pd.DataFrame({"f1": np.ones(n)}, index=df.index)
        fake_y = pd.Series(np.zeros(n), index=df.index)

        mock_model = MagicMock()
        mock_model.predict.return_value = pd.Series([0.07])
        mock_mm = MagicMock()
        mock_mm.load_model.return_value = mock_model

        with patch(
            "src.models.predict_single_stock.add_technical_indicators", return_value=df
        ), patch(
            "src.models.predict_single_stock.create_basic_lag_features",
            return_value=(fake_X, fake_y),
        ), patch(
            "src.models.predict_single_stock.ModelManager", return_value=mock_mm
        ):
            result = _run_single_model_prediction("p/m.joblib", "us", "AAPL", df, 100.0)
        _, pred_return = result
        self.assertAlmostEqual(pred_return, 0.07)


if __name__ == "__main__":
    unittest.main()
