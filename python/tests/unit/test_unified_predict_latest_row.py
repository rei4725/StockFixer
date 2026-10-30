"""stock_features の最新日の行と、統合モデルの予測がその行を使うことのテスト"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.market_data.pipeline import _build_latest_feature_row, fetch_stock_data_with_features
from src.market_data.technical import create_basic_lag_features


def _make_ohlcv(periods: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=periods, freq="B")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, periods))
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": rng.integers(1000, 5000, periods).astype(float),
        },
        index=dates,
    )


def _fetch_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """外部取得をモックして fetch_stock_data_with_features の data を返す。"""
    with (
        patch("src.market_data.pipeline.get_raw_ohlcv_from_db", return_value=None),
        patch("src.market_data.pipeline.should_fetch_fresh_data", return_value=True),
        patch("src.market_data.pipeline.get_stock_data", return_value=ohlcv),
        patch("src.market_data.pipeline.save_raw_ohlcv"),
        patch("src.market_data.pipeline.fetch_cross_asset_features", return_value=None),
        patch("src.market_data.pipeline.fetch_additional_macro_features", return_value=None),
        patch("src.market_data.pipeline.fetch_news_sentiment_with_llm", return_value=None),
        patch("src.market_data.pipeline.fetch_pageview_features", return_value=None),
    ):
        result = fetch_stock_data_with_features(
            market="us", symbol="AAPL", start_date="2023-01-01", end_date="2023-04-30"
        )
    assert result is not None
    return result[2]


class TestStockFeaturesLatestRow(unittest.TestCase):
    def test_data_has_last_ohlcv_day_with_nan_y(self):
        ohlcv = _make_ohlcv(60)
        data = _fetch_features(ohlcv)

        self.assertEqual(data.index[-1], ohlcv.index[-1])
        self.assertTrue(pd.isna(data["y"].iloc[-1]))
        # 最新日の行より前は y が確定している
        self.assertFalse(data["y"].iloc[:-1].isna().any())


class TestBuildLatestFeatureRow(unittest.TestCase):
    def test_matches_row_built_once_next_day_exists(self):
        df = _make_ohlcv(31)
        X_today, _ = create_basic_lag_features(df.iloc[:-1], n_lags=10)
        X_next, _ = create_basic_lag_features(df, n_lags=10)

        latest = _build_latest_feature_row(df.iloc[:-1], list(X_today.columns))

        self.assertIsNotNone(latest)
        self.assertEqual(latest.index[0], df.index[-2])
        pd.testing.assert_frame_equal(
            latest, X_next.loc[[df.index[-2]]], check_dtype=False, check_freq=False
        )

    def test_skips_earnings_day(self):
        df = _make_ohlcv(30)
        df["earnings_flag"] = 0
        df.iloc[-1, df.columns.get_loc("earnings_flag")] = 1
        X, _ = create_basic_lag_features(df, n_lags=10)

        self.assertIsNone(_build_latest_feature_row(df, list(X.columns)))

    def test_skips_when_last_day_has_nan(self):
        df = _make_ohlcv(30)
        X, _ = create_basic_lag_features(df, n_lags=10)
        df.iloc[-1, df.columns.get_loc("Volume")] = np.nan

        self.assertIsNone(_build_latest_feature_row(df, list(X.columns)))


class TestPredictUsesLatestRow(unittest.TestCase):
    def test_predict_uses_last_ohlcv_day_row(self):
        from src.prediction.predict_unified import predict_with_unified_model

        ohlcv = _make_ohlcv(60)
        data = _fetch_features(ohlcv)
        # DB 保存時は market / symbol を列から外して読み戻す
        stored = data.drop(columns=["market", "symbol"])

        mock_model = MagicMock()
        mock_model.predict.return_value = pd.Series([0.01])
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": [100.0]})

        with (
            patch("src.prediction.predict_unified.load_feature_data", return_value=stored),
            patch("src.prediction.predict_unified.get_cached_model", return_value=mock_model),
            patch("src.prediction.predict_unified.get_service_url", return_value=None),
            patch("src.prediction.predict_unified.get_ticker", return_value="AAPL"),
            patch("src.prediction.predict_unified.load_model_weights", return_value=[1.0]),
            patch("src.prediction.predict_unified.yf.Ticker", return_value=mock_ticker),
        ):
            result = predict_with_unified_model("us", "AAPL", model_types=["UnifiedStockXGBoost"])

        self.assertIsNotNone(result)
        passed = mock_model.predict.call_args[0][0]
        self.assertEqual(passed.index[0], ohlcv.index[-1])
        expected = stored.drop(columns=["y"]).iloc[-1]
        pd.testing.assert_series_equal(passed.iloc[0][expected.index], expected, check_names=False)

    def test_predict_with_model_expecting_corporate_action_lags(self):
        """配当・分割のラグの列で学習した今のモデルでも、列を外した特徴量で予想が出る（#816）"""
        from src.prediction.predict_unified import predict_with_unified_model

        ohlcv = _make_ohlcv(60)
        ohlcv["Dividends"] = 0.0
        ohlcv["Stock Splits"] = 0.0
        data = _fetch_features(ohlcv)
        stored = data.drop(columns=["market", "symbol"])
        self.assertNotIn("Dividends_lag1", stored.columns)
        action_lags = [
            f"{col}_lag{n}" for col in ("Dividends", "Stock_Splits") for n in range(1, 11)
        ]
        expected_features = [c for c in stored.columns if c != "y"] + action_lags

        mock_model = MagicMock()
        mock_model.predict.return_value = pd.Series([0.01])
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": [100.0]})

        with (
            patch("src.prediction.predict_unified.load_feature_data", return_value=stored),
            patch("src.prediction.predict_unified.get_cached_model", return_value=mock_model),
            patch(
                "src.prediction.predict_unified.resolve_expected_features",
                return_value=expected_features,
            ),
            patch("src.prediction.predict_unified.get_service_url", return_value=None),
            patch("src.prediction.predict_unified.get_ticker", return_value="AAPL"),
            patch("src.prediction.predict_unified.load_model_weights", return_value=[1.0]),
            patch("src.prediction.predict_unified.yf.Ticker", return_value=mock_ticker),
        ):
            result = predict_with_unified_model("us", "AAPL", model_types=["UnifiedStockXGBoost"])

        self.assertIsNotNone(result)
        passed = mock_model.predict.call_args[0][0]
        self.assertEqual(list(passed.columns), expected_features)
        self.assertTrue((passed[action_lags] == 0).all().all())

    def test_predict_fills_nan_in_model_columns_and_records(self):
        """モデルが持つ列が表にあって値が NaN のとき、0で埋めて記録する（#1425）"""
        from src.prediction.predict_unified import predict_with_unified_model
        from src.prediction.prediction_run_stats import collect_prediction_run_stats

        ohlcv = _make_ohlcv(60)
        data = _fetch_features(ohlcv)
        stored = data.drop(columns=["market", "symbol"]).copy()
        # 特徴量の生成が作らなくなった列は、DB に NULL のまま残る（#816 と同じ形）
        stored["Dividends_lag1"] = np.nan
        expected_features = [c for c in stored.columns if c != "y"]
        filled_col = expected_features[0]
        original_value = stored[filled_col].iloc[-1]
        stored.iloc[-1, stored.columns.get_loc(filled_col)] = np.nan

        mock_model = MagicMock()
        mock_model.predict.return_value = pd.Series([0.01])
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": [100.0]})

        with (
            patch("src.prediction.predict_unified.load_feature_data", return_value=stored),
            patch("src.prediction.predict_unified.get_cached_model", return_value=mock_model),
            patch(
                "src.prediction.predict_unified.resolve_expected_features",
                return_value=expected_features,
            ),
            patch("src.prediction.predict_unified.get_service_url", return_value=None),
            patch("src.prediction.predict_unified.get_ticker", return_value="AAPL"),
            patch("src.prediction.predict_unified.load_model_weights", return_value=[1.0]),
            patch("src.prediction.predict_unified.yf.Ticker", return_value=mock_ticker),
            collect_prediction_run_stats("test") as stats,
        ):
            result = predict_with_unified_model("us", "AAPL", model_types=["UnifiedStockXGBoost"])

        self.assertIsNotNone(result)
        self.assertFalse(pd.isna(original_value))
        passed = mock_model.predict.call_args[0][0]
        self.assertEqual(list(passed.columns), expected_features)
        self.assertFalse(passed.isna().any().any())
        self.assertEqual(passed["Dividends_lag1"].iloc[0], 0)
        self.assertEqual(passed[filled_col].iloc[0], 0)
        # 埋めていない列の値は変わらない
        others = [c for c in expected_features if c not in ("Dividends_lag1", filled_col)]
        pd.testing.assert_series_equal(
            passed.iloc[0][others], stored.iloc[-1][others], check_names=False
        )
        counts = stats.snapshot()[("us", 1)]
        self.assertEqual(counts.filled_symbols, 1)
        self.assertEqual(set(counts.filled_features), {"Dividends_lag1", filled_col})


class TestReadersDropUndeterminedY(unittest.TestCase):
    def _stored_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=4, freq="B"),
                "Close_lag1": [1.0, 2.0, 3.0, 4.0],
                "feature_a": [0.1, 0.2, 0.3, 0.4],
                "y": [0.01, 0.02, 0.03, np.nan],
            }
        )
        return df

    def test_training_loader_drops_nan_y(self):
        from src.prediction.training_pipeline._features import load_features_for_training

        with (
            patch(
                "src.prediction.training_pipeline._features.load_stock_features",
                return_value=self._stored_frame(),
            ),
            patch(
                "src.prediction.training_pipeline._features._mask_earnings_rows",
                side_effect=lambda df, market, symbol: df,
            ),
            patch(
                "src.prediction.training_pipeline._features._apply_feature_exclusions",
                side_effect=lambda X, market, symbol: X,
            ),
        ):
            result = load_features_for_training("us", "AAPL", horizon=1)

        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.y), 3)
        self.assertFalse(result.y.isna().any())
        self.assertEqual(len(result.X), 3)

    def test_unified_training_drops_nan_y(self):
        from src.prediction.unified_model_pipeline import prepare_unified_features

        df = self._stored_frame().drop(columns=["date"])
        df["market"] = "us"
        df["symbol"] = "AAPL"
        with patch(
            "src.prediction.unified_model_pipeline.load_excluded_features", return_value=set()
        ):
            X, y = prepare_unified_features(df)

        self.assertEqual(len(y), 3)
        self.assertFalse(y.isna().any())

    def test_backtest_file_source_drops_nan_y(self):
        from src.backtest.pipeline.features import load_features

        with (
            patch("src.utils.db.load_stock_features", return_value=self._stored_frame()),
            patch("src.market_data.loader.get_raw_ohlcv_from_db", return_value=None),
        ):
            df = load_features("us", "AAPL", "file")

        self.assertEqual(len(df), 3)
        self.assertFalse(df["y"].isna().any())

    def test_backtester_db_loader_drops_nan_y(self):
        from src.market_data.loader import get_stock_data_from_db

        with patch("src.market_data.loader.load_stock_features", return_value=self._stored_frame()):
            df = get_stock_data_from_db("us", "AAPL")

        self.assertEqual(len(df), 3)
        self.assertFalse(df["y"].isna().any())


if __name__ == "__main__":
    unittest.main()
