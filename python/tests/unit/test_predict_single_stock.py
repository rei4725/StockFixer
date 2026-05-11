"""predict_single_stock モジュールのユニットテスト"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.prediction.types import PredictionResult

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
        from src.prediction.predict_single import _resolve_model_types

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
        from src.prediction.predict_single import _fetch_current_price

        mock_hist = pd.DataFrame({"Close": [123.45]}) if yf_hist is None else yf_hist
        mock_ticker = MagicMock()
        if yf_raises:
            mock_ticker.history.side_effect = RuntimeError("network error")
        else:
            mock_ticker.history.return_value = mock_hist

        with patch("src.prediction.predict_single.yf.Ticker", return_value=mock_ticker), patch(
            "src.prediction.predict_single.get_ticker", return_value="AAPL"
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
        from src.prediction.predict_single import _fetch_current_price

        df = pd.DataFrame({"Open": [100.0]})  # Closeなし

        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = RuntimeError("err")
        with patch("src.prediction.predict_single.yf.Ticker", return_value=mock_ticker), patch(
            "src.prediction.predict_single.get_ticker", return_value="AAPL"
        ):
            result = _fetch_current_price("us", "AAPL", df)
        self.assertIsNone(result)


# ──────────────────────────────────────────────────────────────────
# _build_prediction_result  （純粋関数）
# ──────────────────────────────────────────────────────────────────


class TestBuildPredictionResult(unittest.TestCase):
    def _fn(self, market, symbol, current, prices, returns):
        from src.prediction.predict_single import _build_prediction_result

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

    def test_pred_lower_10_and_upper_90_are_set(self):
        """pred_lower_10 / pred_upper_90 が設定されること"""
        result = self._fn("us", "AAPL", 100.0, [105.0], [0.05])
        self.assertIsNotNone(result.pred_lower_10)
        self.assertIsNotNone(result.pred_upper_90)

    def test_interval_ordering(self):
        """pred_lower_10 < avg_pred_price < pred_upper_90 であること"""
        result = self._fn("us", "AAPL", 100.0, [105.0], [0.05])
        self.assertLess(result.pred_lower_10, result.avg_pred_price)
        self.assertGreater(result.pred_upper_90, result.avg_pred_price)

    def test_interval_wider_with_higher_std(self):
        """モデル間分散が大きいほど信頼区間が広くなること"""
        result_narrow = self._fn("us", "AAPL", 100.0, [105.0, 104.0], [0.05, 0.04])
        result_wide = self._fn("us", "AAPL", 100.0, [120.0, 80.0], [0.20, -0.20])
        width_narrow = result_narrow.pred_upper_90 - result_narrow.pred_lower_10
        width_wide = result_wide.pred_upper_90 - result_wide.pred_lower_10
        self.assertGreater(width_wide, width_narrow)

    def test_single_model_uses_default_uncertainty(self):
        """1モデルのとき std=0 でも信頼区間が設定されること"""
        result = self._fn("us", "AAPL", 100.0, [105.0], [0.05])
        # デフォルト 2% の不確実性が適用されるので幅 > 0
        self.assertGreater(result.pred_upper_90 - result.pred_lower_10, 0.0)

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
        from src.prediction.predict_single import _run_single_model_prediction

        n = len(df)
        fake_X = pd.DataFrame({"f1": np.ones(n)}, index=df.index)
        fake_y = pd.Series(np.zeros(n), index=df.index)

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([pred_return])
        mock_mm = MagicMock()
        mock_mm.load_model.return_value = mock_model

        with patch(
            "src.prediction.predict_single.add_technical_indicators", return_value=df
        ), patch(
            "src.prediction.predict_single.create_basic_lag_features",
            return_value=(fake_X, fake_y),
        ), patch(
            "src.prediction.predict_single.ModelManager", return_value=mock_mm
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
        from src.prediction.predict_single import _run_single_model_prediction

        df = _make_price_df(5)
        empty_X = pd.DataFrame()
        empty_y = pd.Series(dtype=float)

        with patch(
            "src.prediction.predict_single.add_technical_indicators", return_value=df
        ), patch(
            "src.prediction.predict_single.create_basic_lag_features",
            return_value=(empty_X, empty_y),
        ), patch(
            "src.prediction.predict_single.ModelManager"
        ):
            result = _run_single_model_prediction("path/model.joblib", "us", "AAPL", df, 100.0)
        self.assertIsNone(result)

    def test_handles_series_prediction(self):
        """model.predict が pd.Series を返すケース"""
        from src.prediction.predict_single import _run_single_model_prediction

        df = _make_price_df()
        n = len(df)
        fake_X = pd.DataFrame({"f1": np.ones(n)}, index=df.index)
        fake_y = pd.Series(np.zeros(n), index=df.index)

        mock_model = MagicMock()
        mock_model.predict.return_value = pd.Series([0.07])
        mock_mm = MagicMock()
        mock_mm.load_model.return_value = mock_model

        with patch(
            "src.prediction.predict_single.add_technical_indicators", return_value=df
        ), patch(
            "src.prediction.predict_single.create_basic_lag_features",
            return_value=(fake_X, fake_y),
        ), patch(
            "src.prediction.predict_single.ModelManager", return_value=mock_mm
        ):
            result = _run_single_model_prediction("p/m.joblib", "us", "AAPL", df, 100.0)
        _, pred_return = result
        self.assertAlmostEqual(pred_return, 0.07)


# ──────────────────────────────────────────────────────────────────
# predict_single_stock  （os.path.exists / data_loader をモック）
# ──────────────────────────────────────────────────────────────────


class TestPredictSingleStock(unittest.TestCase):
    """predict_single_stock のテスト"""

    def _make_df(self, n=50):
        return pd.DataFrame(
            {
                "Close": np.linspace(1000, 1100, n),
                "Open": np.linspace(1000, 1100, n),
                "High": np.linspace(1010, 1110, n),
                "Low": np.linspace(990, 1090, n),
                "Volume": np.ones(n) * 1000,
            }
        )

    @patch("src.prediction.predict_single.load_model_weights")
    @patch("src.prediction.predict_single._run_single_model_prediction")
    @patch("src.prediction.predict_single._fetch_current_price")
    @patch("src.prediction.predict_single.data_loader")
    @patch("src.prediction.predict_single.os.path.exists")
    def test_returns_prediction_result_when_models_exist(
        self, mock_exists, mock_loader, mock_price, mock_run, mock_weights
    ):
        """モデルが存在する場合、PredictionResult が返ること"""
        from src.prediction.predict_single import predict_single_stock

        mock_exists.return_value = True
        mock_loader.get_stock_data.return_value = self._make_df()
        mock_price.return_value = 1100.0
        mock_run.return_value = (1150.0, 0.045)
        mock_weights.return_value = [0.5, 0.5]

        result = predict_single_stock("jp", "7203")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, PredictionResult)
        self.assertEqual(result.symbol, "7203")
        self.assertEqual(result.market, "jp")

    @patch("src.prediction.predict_single.os.path.exists")
    def test_returns_none_when_no_models(self, mock_exists):
        """モデルが存在しない場合 None が返ること"""
        from src.prediction.predict_single import predict_single_stock

        mock_exists.return_value = False
        result = predict_single_stock("jp", "7203")
        self.assertIsNone(result)

    @patch("src.prediction.predict_single.data_loader")
    @patch("src.prediction.predict_single.os.path.exists")
    def test_returns_none_when_features_empty(self, mock_exists, mock_loader):
        """特徴量データが空の場合 None が返ること"""
        from src.prediction.predict_single import predict_single_stock

        mock_exists.return_value = True
        mock_loader.get_stock_data.return_value = pd.DataFrame()
        result = predict_single_stock("jp", "7203")
        self.assertIsNone(result)

    @patch("src.prediction.predict_single._fetch_current_price")
    @patch("src.prediction.predict_single.data_loader")
    @patch("src.prediction.predict_single.os.path.exists")
    def test_returns_none_when_price_none(self, mock_exists, mock_loader, mock_price):
        """現在値が取得できない場合 None が返ること"""
        from src.prediction.predict_single import predict_single_stock

        mock_exists.return_value = True
        mock_loader.get_stock_data.return_value = self._make_df()
        mock_price.return_value = None
        result = predict_single_stock("jp", "7203")
        self.assertIsNone(result)

    @patch("src.prediction.predict_single.load_model_weights")
    @patch("src.prediction.predict_single._run_single_model_prediction")
    @patch("src.prediction.predict_single._fetch_current_price")
    @patch("src.prediction.predict_single.data_loader")
    @patch("src.prediction.predict_single.os.path.exists")
    def test_continues_on_model_exception(
        self, mock_exists, mock_loader, mock_price, mock_run, mock_weights
    ):
        """モデル予測で例外が出ても次のモデルに続くこと"""
        from src.prediction.predict_single import predict_single_stock

        mock_exists.return_value = True
        mock_loader.get_stock_data.return_value = self._make_df()
        mock_price.return_value = 1100.0
        # horizon=1 → 2モデル: XGBoost が例外、LightGBM が成功
        mock_run.side_effect = [Exception("モデルエラー"), (1150.0, 0.045)]
        mock_weights.return_value = [1.0]

        result = predict_single_stock("jp", "7203")
        # 2番目のモデルで成功するので PredictionResult が返る
        self.assertIsNotNone(result)
        self.assertIsInstance(result, PredictionResult)


# ──────────────────────────────────────────────────────────────────
# predict_single_stock_multi_horizon  （predict_single_stock をモック）
# ──────────────────────────────────────────────────────────────────


class TestPredictSingleStockMultiHorizon(unittest.TestCase):
    """predict_single_stock_multi_horizon のテスト"""

    def _make_result(self, diff_ratio=0.05):
        return PredictionResult(
            market="jp",
            symbol="7203",
            current_price=1000.0,
            avg_pred_price=1000.0 * (1 + diff_ratio),
            diff_ratio=diff_ratio,
            model_count=2,
        )

    @patch("src.prediction.predict_single.predict_single_stock")
    def test_returns_prediction_result_with_horizons(self, mock_predict):
        """各ホライズンで成功した場合 PredictionResult が返ること"""
        from src.prediction.predict_single import predict_single_stock_multi_horizon

        mock_predict.return_value = self._make_result()
        result = predict_single_stock_multi_horizon("jp", "7203", horizons=[1, 3, 5])

        self.assertIsNotNone(result)
        self.assertIsInstance(result, PredictionResult)
        self.assertEqual(result.symbol, "7203")

    @patch("src.prediction.predict_single.predict_single_stock")
    def test_returns_none_when_all_horizons_fail(self, mock_predict):
        """全ホライズンが None を返した場合 None が返ること"""
        from src.prediction.predict_single import predict_single_stock_multi_horizon

        mock_predict.return_value = None
        result = predict_single_stock_multi_horizon("jp", "7203", horizons=[1, 3, 5])
        self.assertIsNone(result)

    @patch("src.prediction.predict_single.predict_single_stock")
    def test_multi_horizon_fields_populated(self, mock_predict):
        """3d/5d/10d フィールドが結果に含まれること"""
        from src.prediction.predict_single import predict_single_stock_multi_horizon

        def _side(market, symbol, horizon=1, **kwargs):
            return self._make_result(diff_ratio=0.05 * horizon)

        mock_predict.side_effect = _side
        result = predict_single_stock_multi_horizon("jp", "7203", horizons=[1, 3, 5, 10])

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.avg_pred_price_3d)
        self.assertIsNotNone(result.avg_pred_price_5d)
        self.assertIsNotNone(result.avg_pred_price_10d)

    @patch("src.prediction.predict_single.predict_single_stock")
    def test_confluence_score_all_upward(self, mock_predict):
        """全ホライズンが上昇予測のとき confluence_score がホライズン数と一致すること"""
        from src.prediction.predict_single import predict_single_stock_multi_horizon

        mock_predict.return_value = self._make_result(diff_ratio=0.05)
        result = predict_single_stock_multi_horizon("jp", "7203", horizons=[1, 3, 5])

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.confluence_score)
        self.assertGreaterEqual(result.confluence_score, 0)
        self.assertEqual(result.confluence_score, 3)  # 3ホライズン全て一致

    @patch("src.prediction.predict_single.predict_single_stock")
    def test_horizons_dict_populated(self, mock_predict):
        """horizons dict に各ホライズンの HorizonResult が格納されること"""
        from src.prediction.predict_single import predict_single_stock_multi_horizon
        from src.prediction.types import HorizonResult

        def _side(market, symbol, horizon=1, **kwargs):
            return self._make_result(diff_ratio=0.05 * horizon)

        mock_predict.side_effect = _side
        result = predict_single_stock_multi_horizon("jp", "7203", horizons=[1, 3, 5, 10])

        self.assertIsNotNone(result)
        self.assertIn(1, result.horizons)
        self.assertIn(3, result.horizons)
        self.assertIn(5, result.horizons)
        self.assertIn(10, result.horizons)
        self.assertIsInstance(result.horizons[3], HorizonResult)
        self.assertEqual(result.horizons[3].horizon_days, 3)

    @patch("src.prediction.predict_single.predict_single_stock")
    def test_base_horizon_is_1_when_available(self, mock_predict):
        """horizon=1 の結果が base として使われること"""
        from src.prediction.predict_single import predict_single_stock_multi_horizon

        def _side(market, symbol, horizon=1, **kwargs):
            return self._make_result(diff_ratio=0.01 * horizon)

        mock_predict.side_effect = _side
        result = predict_single_stock_multi_horizon("jp", "7203", horizons=[1, 3])

        self.assertIsNotNone(result)
        # horizon=1 の diff_ratio は 0.01
        self.assertAlmostEqual(result.diff_ratio, 0.01)


if __name__ == "__main__":
    unittest.main()
