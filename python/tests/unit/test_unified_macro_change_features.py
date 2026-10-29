"""統合モデルの学習で相場全体の水準の列を外し、前日比の変化率の列を使うことのテスト（#1374）"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.market_data.technical import create_basic_lag_features

_MACROS = ("sp500_close", "gold_close", "dxy_close", "usdjpy_close", "tnx_close", "vix_close")
_N_LAGS = 10


def _stored_features(periods: int = 40, seed: int = 0) -> pd.DataFrame:
    """stock_features と同じ形（生の値の列は無く、ラグの列だけ）の表を作る。"""
    dates = pd.date_range("2024-01-01", periods=periods, freq="B")
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame(
        {"Close": 100 + np.cumsum(rng.normal(0, 1, periods))},
        index=dates,
    )
    for i, name in enumerate(_MACROS):
        raw[name] = 50 + 10 * i + np.cumsum(rng.normal(0, 1, periods))
    X, y = create_basic_lag_features(raw, n_lags=_N_LAGS)
    stored = X.copy()
    stored["y"] = y
    stored["market"] = "us"
    stored["symbol"] = "AAPL"
    stored["date"] = stored.index
    return stored.reset_index(drop=True)


def _prepare(df: pd.DataFrame):
    from src.prediction.unified_model_pipeline import prepare_unified_features

    with patch("src.prediction.unified_model_pipeline.load_excluded_features", return_value=set()):
        return prepare_unified_features(df)


class TestTrainingFeatureColumns(unittest.TestCase):
    def test_level_columns_are_not_features(self):
        X, _ = _prepare(_stored_features())
        for name in _MACROS:
            for lag in range(1, _N_LAGS + 1):
                self.assertNotIn(f"{name}_lag{lag}", X.columns)

    def test_change_columns_are_features(self):
        X, _ = _prepare(_stored_features())
        for name in _MACROS:
            for lag in range(1, _N_LAGS):
                self.assertIn(f"{name}_pct_lag{lag}", X.columns)
        # Close のラグは銘柄ごとの列なので残す
        self.assertIn("Close_lag1", X.columns)

    def test_raw_level_column_is_not_feature(self):
        df = _stored_features()
        df["sp500_close"] = 1.0
        X, _ = _prepare(df)
        self.assertNotIn("sp500_close", X.columns)


class TestChangeColumnsDoNotLookAhead(unittest.TestCase):
    def test_change_at_t_uses_values_up_to_t_minus_1(self):
        periods = 40
        dates = pd.date_range("2024-01-01", periods=periods, freq="B")
        level = pd.Series(np.arange(1, periods + 1, dtype=float) ** 1.5, index=dates)
        raw = pd.DataFrame({"Close": np.linspace(100, 120, periods), "sp500_close": level})
        X_lag, y = create_basic_lag_features(raw, n_lags=_N_LAGS)
        df = X_lag.copy()
        df["y"] = y
        df["market"] = "us"
        df["symbol"] = "AAPL"
        X, _ = _prepare(df.reset_index(drop=True))

        expected = (level.shift(1) / level.shift(2) - 1).loc[X_lag.index].to_numpy()
        np.testing.assert_allclose(X["sp500_close_pct_lag1"], expected)

    def test_change_at_t_does_not_move_with_value_at_t(self):
        periods = 40
        dates = pd.date_range("2024-01-01", periods=periods, freq="B")
        level = pd.Series(np.linspace(10, 50, periods), index=dates)
        bumped = level.copy()
        bumped.iloc[-2] *= 3.0  # t の当日の値だけ変える（最後の行は y が無く落ちる）

        def build(series: pd.Series) -> pd.DataFrame:
            raw = pd.DataFrame({"Close": np.linspace(100, 120, periods), "sp500_close": series})
            X_lag, y = create_basic_lag_features(raw, n_lags=_N_LAGS)
            df = X_lag.copy()
            df["y"] = y
            df["market"] = "us"
            df["symbol"] = "AAPL"
            X, _ = _prepare(df.reset_index(drop=True))
            return X

        base, moved = build(level), build(bumped)
        cols = [f"sp500_close_pct_lag{k}" for k in range(1, _N_LAGS)]
        # 当日 t の行の変化率は t の値で動かない
        pd.testing.assert_series_equal(base[cols].iloc[-1], moved[cols].iloc[-1])


def _predict(stored: pd.DataFrame, expected_features: list[str]):
    from src.prediction.predict_unified import predict_with_unified_model

    mock_model = MagicMock()
    mock_model.predict.return_value = pd.Series([0.01])
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame(
        {"Close": [100.0]}, index=pd.DatetimeIndex([stored["date"].iloc[-1]])
    )
    zero_fill = MagicMock()
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
        patch("src.prediction.predict_unified.count_weekdays_behind", return_value=0),
        patch("src.prediction.predict_unified._fetch_current_price", return_value=100.0),
        patch("src.prediction.predict_unified.record_zero_fill", zero_fill),
    ):
        result = predict_with_unified_model("us", "AAPL", model_types=["UnifiedStockXGBoost"])
    return result, mock_model.predict.call_args[0][0], zero_fill.call_args[0][2]


class TestPredictionPath(unittest.TestCase):
    def test_production_model_with_level_columns_predicts_without_zero_fill(self):
        """今の本番のモデル（水準の列で学習）は、表の列のまま0埋めなしで予想する"""
        stored = _stored_features().drop(columns=["market", "symbol"])
        expected_features = [c for c in stored.columns if c not in ("y", "date")]
        self.assertIn("sp500_close_lag1", expected_features)

        result, passed, zero_filled = _predict(stored, expected_features)

        self.assertIsNotNone(result)
        self.assertEqual(list(passed.columns), expected_features)
        self.assertEqual(zero_filled, [])
        pd.testing.assert_series_equal(
            passed.iloc[0], stored[expected_features].iloc[-1], check_names=False
        )

    def test_challenger_model_with_change_columns_predicts_without_zero_fill(self):
        """変化率の列で学習した challenger も、予想の道で0埋めなしに列がそろう"""
        stored = _stored_features().drop(columns=["market", "symbol"])
        train_df = _stored_features()
        X_train, _ = _prepare(train_df)
        expected_features = list(X_train.columns)

        result, passed, zero_filled = _predict(stored, expected_features)

        self.assertIsNotNone(result)
        self.assertEqual(list(passed.columns), expected_features)
        self.assertEqual(zero_filled, [])
        last = stored.iloc[-1]
        self.assertAlmostEqual(
            float(passed["sp500_close_pct_lag1"].iloc[0]),
            last["sp500_close_lag1"] / last["sp500_close_lag2"] - 1,
        )


if __name__ == "__main__":
    unittest.main()
