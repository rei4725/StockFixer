"""特徴量が古いかの判定が、jp では東証の休場日を数えないことのテスト（#1106）。

2027-05-06（木）の朝、特徴量の最後の行は 2027-04-30（金）になる。05-03〜05-05 は
東証の休場日のため、足りない営業日は 05-06 の1日だけである。
"""

import unittest
from contextlib import ExitStack
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.prediction.predict_unified import count_weekdays_behind
from src.prediction.types import PredictionResult

_LAST_ROW = pd.Timestamp("2027-04-30")
_TODAY = datetime(2027, 5, 6, 7, 30)


def _fixed_now() -> MagicMock:
    """predict_unified の datetime.now() を _TODAY に固定する。"""
    mock_datetime = MagicMock()
    mock_datetime.now.return_value = _TODAY
    return mock_datetime


class TestCountWeekdaysBehindJpHolidays(unittest.TestCase):
    def test_jp_skips_golden_week_holidays(self):
        self.assertEqual(count_weekdays_behind(_LAST_ROW, _TODAY, market="jp"), 1)

    def test_us_counts_weekdays_only(self):
        self.assertEqual(count_weekdays_behind(_LAST_ROW, _TODAY, market="us"), 4)

    def test_market_omitted_counts_weekdays_only(self):
        self.assertEqual(count_weekdays_behind(_LAST_ROW, _TODAY), 4)

    def test_jp_expected_latest_skips_holiday_today(self):
        today = datetime(2027, 5, 5, 7, 30)
        self.assertEqual(count_weekdays_behind(_LAST_ROW, today, market="jp"), 0)

    def test_jp_stale_by_business_days(self):
        last_row = pd.Timestamp("2027-04-23")
        # 04-26・04-27・04-28・04-30・05-06 の5営業日（04-29 は休場）
        self.assertEqual(count_weekdays_behind(last_row, _TODAY, market="jp"), 5)


def _feature_frame(last_date: pd.Timestamp, n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.bdate_range(end=last_date, periods=n),
            "y": np.linspace(0.001, 0.01, n),
            "Close_lag1": np.linspace(998, 1098, n),
            "feature_a": np.linspace(-1, 1, n),
        }
    )


def _unified_model() -> MagicMock:
    model = MagicMock()
    model.model = MagicMock(feature_names_in_=["Close_lag1", "feature_a"])
    model.predict.return_value = pd.Series([0.01])
    return model


class TestPredictUnifiedJpHolidays(unittest.TestCase):
    def setUp(self):
        from src.prediction import predict_unified

        predict_unified._model_cache.clear()

    def _run(self, last_date: pd.Timestamp):
        from src.prediction.predict_unified import predict_with_unified_model

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": [1050.0]})
        on_stale = MagicMock()
        with ExitStack() as stack:
            for p in (
                patch("src.prediction.predict_unified.datetime", _fixed_now()),
                patch(
                    "src.prediction.predict_unified.load_feature_data",
                    return_value=_feature_frame(last_date),
                ),
                patch(
                    "src.prediction.predict_unified.get_cached_model",
                    return_value=_unified_model(),
                ),
                patch("src.prediction.predict_unified.get_ticker", return_value="5202.T"),
                patch("src.prediction.predict_unified.load_model_weights", return_value=[0.5, 0.5]),
                patch("src.prediction.predict_unified.get_service_url", return_value=None),
                patch("src.prediction.predict_unified.yf.Ticker", return_value=mock_ticker),
            ):
                stack.enter_context(p)
            result = predict_with_unified_model("jp", "5202", on_stale=on_stale)
        return result, on_stale

    def test_predicts_after_golden_week(self):
        result, on_stale = self._run(_LAST_ROW)

        self.assertIsNotNone(result)
        on_stale.assert_not_called()

    def test_skips_when_stale_by_business_days(self):
        result, on_stale = self._run(pd.Timestamp("2027-04-23"))

        self.assertIsNone(result)
        on_stale.assert_called_once_with("jp", "5202")


def _ohlcv(last_date: pd.Timestamp, n: int = 60) -> pd.DataFrame:
    close = np.linspace(1000, 1100, n)
    return pd.DataFrame(
        {
            "Open": close - 5,
            "High": close + 10,
            "Low": close - 10,
            "Close": close,
            "Volume": np.full(n, 500_000, dtype=float),
        },
        index=pd.bdate_range(end=last_date, periods=n),
    )


class TestPredictSingleJpHolidays(unittest.TestCase):
    def _predict(self, last_date: pd.Timestamp):
        from src.market_data.technical import create_basic_lag_features
        from src.prediction.predict_single import predict_single_stock

        mock_port = MagicMock()
        mock_port.get_stock_data.return_value = _ohlcv(last_date)
        mock_port.add_technical_indicators.side_effect = lambda d: d.copy()
        mock_port.fetch_cross_asset_features.return_value = None
        mock_port.create_basic_lag_features.side_effect = create_basic_lag_features
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.01])
        mock_mm = MagicMock()
        mock_mm.load_model.return_value = mock_model
        with (
            patch("src.prediction.predict_unified.datetime", _fixed_now()),
            patch("src.prediction.predict_single.os.path.exists", return_value=True),
            patch("src.prediction.predict_single.get_market_data_port", return_value=mock_port),
            patch("src.prediction.predict_single.ModelManager", return_value=mock_mm),
            patch("src.prediction.predict_single._fetch_current_price", return_value=1100.0),
            patch("src.prediction.predict_single.load_model_weights", return_value=[0.5, 0.5]),
        ):
            return predict_single_stock("jp", "5202")

    def test_predicts_after_golden_week(self):
        self.assertIsInstance(self._predict(_LAST_ROW), PredictionResult)

    def test_skips_when_stale_by_business_days(self):
        self.assertIsNone(self._predict(pd.Timestamp("2027-04-23")))


if __name__ == "__main__":
    unittest.main()
