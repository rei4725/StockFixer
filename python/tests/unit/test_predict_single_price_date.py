"""銘柄別モデルの予想の基準の値段が、予想に使った足と同じ日の終値であることのテスト（#867）"""

import logging
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.prediction.types import PredictionResult


def _latest_weekday() -> pd.Timestamp:
    today = pd.Timestamp.today().normalize()
    while today.weekday() >= 5:
        today -= pd.Timedelta(days=1)
    return today


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


def _make_hist(bar_date: pd.Timestamp, close: float) -> pd.DataFrame:
    idx = pd.DatetimeIndex([bar_date]).tz_localize("Asia/Tokyo")
    return pd.DataFrame({"Close": [close]}, index=idx)


def _make_port(df: pd.DataFrame) -> MagicMock:
    from src.market_data.technical import create_basic_lag_features

    mock_port = MagicMock()
    mock_port.get_stock_data.return_value = df
    mock_port.add_technical_indicators.side_effect = lambda d: d.copy()
    mock_port.fetch_cross_asset_features.return_value = None
    mock_port.create_basic_lag_features.side_effect = create_basic_lag_features
    return mock_port


class TestPredictSingleStockPriceDate(unittest.TestCase):
    def _predict(self, df: pd.DataFrame, hists: list):
        from src.prediction.predict_single import predict_single_stock

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.01])
        mock_mm = MagicMock()
        mock_mm.load_model.return_value = mock_model
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = hists
        with (
            patch("src.prediction.predict_single.os.path.exists", return_value=True),
            patch(
                "src.prediction.predict_single.get_market_data_port", return_value=_make_port(df)
            ),
            patch("src.prediction.predict_single.ModelManager", return_value=mock_mm),
            patch("src.prediction.predict_single.yf.Ticker", return_value=mock_ticker),
            patch("src.prediction.predict_single.load_model_weights", return_value=[0.5, 0.5]),
        ):
            result = predict_single_stock("jp", "5202")
        return result, mock_ticker

    def test_returns_none_and_warns_when_bar_date_differs(self):
        last_date = _latest_weekday()
        bar_date = last_date - pd.Timedelta(days=1)
        hist = _make_hist(bar_date, 1100.0)
        with self.assertLogs("src.prediction.predict_single", level=logging.WARNING) as logs:
            result, _ = self._predict(_make_df(last_date), [hist, hist])

        self.assertIsNone(result)
        self.assertTrue(
            any(
                "5202" in line
                and bar_date.strftime("%Y-%m-%d") in line
                and last_date.strftime("%Y-%m-%d") in line
                for line in logs.output
            ),
            logs.output,
        )

    def test_returns_none_and_warns_when_close_is_nan(self):
        last_date = _latest_weekday()
        hist = _make_hist(last_date, float("nan"))
        with self.assertLogs("src.prediction.predict_single", level=logging.WARNING) as logs:
            result, _ = self._predict(_make_df(last_date), [hist, hist])

        self.assertIsNone(result)
        self.assertTrue(
            any(
                "5202" in line and line.count(last_date.strftime("%Y-%m-%d")) >= 2
                for line in logs.output
            ),
            logs.output,
        )

    def test_returns_prediction_when_bar_date_matches(self):
        last_date = _latest_weekday()
        hist = _make_hist(last_date, 1100.0)
        result, _ = self._predict(_make_df(last_date), [hist, hist])

        self.assertIsInstance(result, PredictionResult)
        self.assertEqual(result.model_count, 2)
        self.assertAlmostEqual(result.current_price, 1100.0)
        self.assertAlmostEqual(result.avg_pred_price, 1100.0 * 1.01)

    def test_fetches_current_price_once_for_two_models(self):
        last_date = _latest_weekday()
        hists = [_make_hist(last_date, 1100.0), _make_hist(last_date, 1090.0)]
        result, mock_ticker = self._predict(_make_df(last_date), hists)

        self.assertIsInstance(result, PredictionResult)
        self.assertEqual(result.model_count, 2)
        self.assertEqual(mock_ticker.history.call_count, 1)
        # どちらのモデルも変化率 0.01 なので、基準が同じなら平均は current_price * 1.01
        self.assertAlmostEqual(result.avg_pred_price, result.current_price * 1.01)
        self.assertAlmostEqual(result.diff_ratio, 0.01)


if __name__ == "__main__":
    unittest.main()
