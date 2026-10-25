"""銘柄別モデルの予想が、止まった足で出ないことのテスト（#854）"""

import logging
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from config.settings import STALE_BARS_MAX_WEEKDAYS
from src.domain.jp_market_holidays import is_jp_business_day
from src.prediction.types import PredictionResult


def _latest_weekday() -> pd.Timestamp:
    today = pd.Timestamp.today().normalize()
    while not is_jp_business_day(today.date()):
        today -= pd.Timedelta(days=1)
    return today


def _business_days_before(n: int) -> pd.Timestamp:
    """直近の東証の営業日から東証の営業日で n 日戻った日（#1106）。"""
    day = _latest_weekday()
    for _ in range(n):
        day -= pd.Timedelta(days=1)
        while not is_jp_business_day(day.date()):
            day -= pd.Timedelta(days=1)
    return day


def _make_df(last_date: pd.Timestamp, n: int = 60) -> pd.DataFrame:
    idx = pd.bdate_range(end=last_date, periods=n)
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


def _make_port(df: pd.DataFrame) -> MagicMock:
    from src.market_data.technical import create_basic_lag_features

    mock_port = MagicMock()
    mock_port.get_stock_data.return_value = df
    mock_port.add_technical_indicators.side_effect = lambda d: d.copy()
    mock_port.fetch_cross_asset_features.return_value = None
    mock_port.create_basic_lag_features.side_effect = create_basic_lag_features
    return mock_port


def _make_mm() -> MagicMock:
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.01])
    mock_mm = MagicMock()
    mock_mm.load_model.return_value = mock_model
    return mock_mm


class TestPredictSingleStockStaleBars(unittest.TestCase):
    def _predict(self, df: pd.DataFrame):
        from src.prediction.predict_single import predict_single_stock

        mock_mm = _make_mm()
        with (
            patch("src.prediction.predict_single.os.path.exists", return_value=True),
            patch(
                "src.prediction.predict_single.get_market_data_port", return_value=_make_port(df)
            ),
            patch("src.prediction.predict_single.ModelManager", return_value=mock_mm),
            patch("src.prediction.predict_single._fetch_current_price", return_value=1100.0),
            patch("src.prediction.predict_single.load_model_weights", return_value=[0.5, 0.5]),
        ):
            result = predict_single_stock("jp", "5202")
        return result, mock_mm

    def test_returns_none_and_warns_when_last_bar_is_stale(self):
        last_date = _business_days_before(STALE_BARS_MAX_WEEKDAYS + 1)
        with self.assertLogs("src.prediction.predict_single", level=logging.WARNING) as logs:
            result, mock_mm = self._predict(_make_df(last_date))

        self.assertIsNone(result)
        mock_mm.load_model.assert_not_called()
        self.assertTrue(any("5202" in line for line in logs.output))
        self.assertTrue(any(last_date.strftime("%Y-%m-%d") in line for line in logs.output))

    def test_returns_prediction_when_last_bar_is_within_limit(self):
        last_date = _business_days_before(STALE_BARS_MAX_WEEKDAYS)
        result, _ = self._predict(_make_df(last_date))

        self.assertIsInstance(result, PredictionResult)
        self.assertEqual(result.model_count, 2)

    def test_returns_prediction_when_last_bar_is_today(self):
        result, _ = self._predict(_make_df(_latest_weekday()))

        self.assertIsInstance(result, PredictionResult)

    def test_handles_tz_aware_index(self):
        df = _make_df(_business_days_before(STALE_BARS_MAX_WEEKDAYS + 1))
        df.index = df.index.tz_localize("Asia/Tokyo")
        result, _ = self._predict(df)

        self.assertIsNone(result)


class TestExplainPredictionShapStaleBars(unittest.TestCase):
    def _explain(self, df: pd.DataFrame):
        from src.prediction.predict_single import explain_prediction_shap

        mock_mm = _make_mm()
        mock_shap = MagicMock()
        mock_shap.TreeExplainer.return_value.shap_values.side_effect = lambda X: np.zeros(
            (1, X.shape[1])
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "StockXGBoostModel.joblib"), "wb") as f:
                f.write(b"dummy")
            with (
                patch.dict("sys.modules", {"shap": mock_shap}),
                patch("src.prediction.predict_single.get_models_subdir", return_value=tmpdir),
                patch(
                    "src.prediction.predict_single.get_market_data_port",
                    return_value=_make_port(df),
                ),
                patch("src.prediction.predict_single.ModelManager", return_value=mock_mm),
                patch("src.prediction.predict_single.normalize_col", side_effect=lambda x: x),
            ):
                return explain_prediction_shap("jp", "5202")

    def test_returns_none_and_warns_when_last_bar_is_stale(self):
        last_date = _business_days_before(STALE_BARS_MAX_WEEKDAYS + 1)
        with self.assertLogs("src.prediction.predict_single", level=logging.WARNING) as logs:
            result = self._explain(_make_df(last_date))

        self.assertIsNone(result)
        self.assertTrue(any("5202" in line for line in logs.output))

    def test_returns_result_when_last_bar_is_within_limit(self):
        last_date = _business_days_before(STALE_BARS_MAX_WEEKDAYS)
        result = self._explain(_make_df(last_date))

        self.assertIsNotNone(result)
        self.assertEqual(result["symbol"], "5202")


if __name__ == "__main__":
    unittest.main()
