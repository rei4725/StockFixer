"""predict_with_unified_model の基準の値段の足の日付のテスト（#845）。

基準の値段 current_price は yfinance の最後の足の終値である。その足が特徴量の
最後の行と別の日なら、1日ずれた値段に予想の変化率を掛けることになる。取れない
ときに Close_lag1（前の営業日の終値）や y（変化率）を基準にすることもしない。
"""

from __future__ import annotations

import unittest
from contextlib import ExitStack
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


def _latest_weekday() -> pd.Timestamp:
    """今日から遡った直近の平日（特徴量が古いと判定されない日付）。"""
    day = pd.Timestamp.now().normalize()
    while day.weekday() >= 5:
        day -= pd.Timedelta(days=1)
    return day


def _feature_frame(last_date: pd.Timestamp, n: int = 10, with_close_lag1: bool = True):
    """本番の load_stock_features 相当。date 列は時刻 0 時のタイムゾーン無し。"""
    frame = {
        "date": pd.bdate_range(end=last_date, periods=n),
        "y": np.linspace(0.001, 0.01, n),
        "feature_a": np.linspace(-1, 1, n),
    }
    if with_close_lag1:
        frame["Close_lag1"] = np.linspace(998, 1098, n)
    return pd.DataFrame(frame)


def _history(bar_date: pd.Timestamp, close: float = 1050.0, tz: Optional[str] = None):
    """yfinance の history 相当。index は取引所のタイムゾーンの 0 時。"""
    index = pd.DatetimeIndex([bar_date])
    if tz is not None:
        index = index.tz_localize(tz)
    return pd.DataFrame({"Close": [close]}, index=index)


def _mock_model() -> MagicMock:
    model = MagicMock()
    model.model = MagicMock(feature_names_in_=["Close_lag1", "feature_a"])
    model.predict.return_value = pd.Series([0.01])
    return model


def _run(df: pd.DataFrame, model: MagicMock, ticker: MagicMock, market: str = "jp"):
    from src.prediction.predict_unified import predict_with_unified_model

    with ExitStack() as stack:
        for p in [
            patch("src.prediction.predict_unified.load_feature_data", return_value=df),
            patch("src.prediction.predict_unified.get_cached_model", return_value=model),
            patch("src.prediction.predict_unified.get_ticker", return_value="7203.T"),
            patch("src.prediction.predict_unified.load_model_weights", return_value=[0.5, 0.5]),
            patch("src.prediction.predict_unified.get_service_url", return_value=None),
            patch("src.prediction.predict_unified.yf.Ticker", return_value=ticker),
        ]:
            stack.enter_context(p)
        return predict_with_unified_model(market, "7203")


def _ticker_returning(hist: pd.DataFrame) -> MagicMock:
    ticker = MagicMock()
    ticker.history.return_value = hist
    return ticker


class TestCurrentPriceBarDate(unittest.TestCase):
    def setUp(self):
        from src.prediction import predict_unified

        predict_unified._model_cache.clear()

    def test_bar_date_mismatch_returns_none_and_warns(self):
        feature_date = _latest_weekday()
        bar_date = feature_date - BDay(1)
        model = _mock_model()

        with self.assertLogs("src.prediction.predict_unified", level="WARNING") as logs:
            result = _run(
                _feature_frame(feature_date), model, _ticker_returning(_history(bar_date))
            )

        self.assertIsNone(result)
        model.predict.assert_not_called()
        joined = "\n".join(logs.output)
        self.assertIn("7203", joined)
        self.assertIn(feature_date.strftime("%Y-%m-%d"), joined)
        self.assertIn(bar_date.strftime("%Y-%m-%d"), joined)

    def test_newer_bar_than_features_returns_none(self):
        feature_date = _latest_weekday() - BDay(1)
        bar_date = _latest_weekday()
        model = _mock_model()

        result = _run(_feature_frame(feature_date), model, _ticker_returning(_history(bar_date)))

        self.assertIsNone(result)
        model.predict.assert_not_called()

    def test_empty_history_does_not_fall_back_to_close_lag1(self):
        model = _mock_model()

        with self.assertLogs("src.prediction.predict_unified", level="WARNING"):
            result = _run(
                _feature_frame(_latest_weekday()), model, _ticker_returning(pd.DataFrame())
            )

        self.assertIsNone(result)
        model.predict.assert_not_called()

    def test_history_exception_does_not_fall_back_to_close_lag1(self):
        model = _mock_model()
        ticker = MagicMock()
        ticker.history.side_effect = RuntimeError("network error")

        with self.assertLogs("src.prediction.predict_unified", level="WARNING"):
            result = _run(_feature_frame(_latest_weekday()), model, ticker)

        self.assertIsNone(result)
        model.predict.assert_not_called()

    def test_empty_history_without_close_lag1_does_not_use_y(self):
        model = _mock_model()
        df = _feature_frame(_latest_weekday(), with_close_lag1=False)

        result = _run(df, model, _ticker_returning(pd.DataFrame()))

        self.assertIsNone(result)
        model.predict.assert_not_called()

    def test_matching_date_keeps_prediction_price(self):
        feature_date = _latest_weekday()
        model = _mock_model()

        result = _run(
            _feature_frame(feature_date), model, _ticker_returning(_history(feature_date))
        )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.current_price, 1050.0)
        self.assertAlmostEqual(result.avg_pred_price, 1050.0 * 1.01)
        self.assertAlmostEqual(result.diff_ratio, 0.01)

    def test_matching_date_with_exchange_timezone_jp(self):
        feature_date = _latest_weekday()
        model = _mock_model()

        result = _run(
            _feature_frame(feature_date),
            model,
            _ticker_returning(_history(feature_date, tz="Asia/Tokyo")),
            market="jp",
        )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.current_price, 1050.0)

    def test_matching_date_with_exchange_timezone_us(self):
        feature_date = _latest_weekday()
        model = _mock_model()

        result = _run(
            _feature_frame(feature_date),
            model,
            _ticker_returning(_history(feature_date, tz="America/New_York")),
            market="us",
        )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.current_price, 1050.0)


if __name__ == "__main__":
    unittest.main()
