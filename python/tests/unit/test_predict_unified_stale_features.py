"""predict_with_unified_model が古い特徴量で予想を出さないことのテスト（#829）。

足が止まった銘柄は特徴量の生成でスキップされる（#815）が、stock_features には
止まった日を最後の行とする前の特徴量が残る。予想の側でも最後の行の日付を見る。
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from config.settings import STALE_BARS_MAX_WEEKDAYS
from src.utils.jp_market_holidays import is_jp_business_day


def _latest_weekday() -> pd.Timestamp:
    """今日から遡った直近の東証の営業日（あるべき最新の営業日）。"""
    day = pd.Timestamp.now().normalize()
    while not is_jp_business_day(day.date()):
        day -= pd.Timedelta(days=1)
    return day


def _business_days_before(n: int) -> pd.Timestamp:
    """あるべき最新の営業日から東証の営業日で n 日戻った日（#1106）。"""
    day = _latest_weekday()
    for _ in range(n):
        day -= pd.Timedelta(days=1)
        while not is_jp_business_day(day.date()):
            day -= pd.Timedelta(days=1)
    return day


def _feature_frame(last_date: pd.Timestamp, n: int = 10) -> pd.DataFrame:
    """本番の load_stock_features 相当。date 列の最後が last_date になる。"""
    return pd.DataFrame(
        {
            "date": pd.bdate_range(end=last_date, periods=n),
            "y": np.linspace(0.001, 0.01, n),
            "Close_lag1": np.linspace(998, 1098, n),
            "feature_a": np.linspace(-1, 1, n),
        }
    )


def _mock_model() -> MagicMock:
    model = MagicMock()
    model.model = MagicMock(feature_names_in_=["Close_lag1", "feature_a"])
    model.predict.return_value = pd.Series([0.01])
    return model


def _patches(feature_frame_for, model: MagicMock) -> list:
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame({"Close": [1050.0]})
    return [
        patch("src.prediction.predict_unified.load_feature_data", side_effect=feature_frame_for),
        patch("src.prediction.predict_unified.get_cached_model", return_value=model),
        patch("src.prediction.predict_unified.get_ticker", return_value="5202.T"),
        patch("src.prediction.predict_unified.load_model_weights", return_value=[0.5, 0.5]),
        patch("src.prediction.predict_unified.get_service_url", return_value=None),
        patch("src.prediction.predict_unified.yf.Ticker", return_value=mock_ticker),
    ]


def _run(feature_frame_for, model: MagicMock):
    from contextlib import ExitStack

    from src.prediction.predict_unified import predict_with_unified_model

    with ExitStack() as stack:
        for p in _patches(feature_frame_for, model):
            stack.enter_context(p)
        return predict_with_unified_model("jp", "5202")


class TestStaleFeatures(unittest.TestCase):
    def setUp(self):
        from src.prediction import predict_unified

        predict_unified._model_cache.clear()

    def test_stale_last_row_returns_none_and_warns(self):
        last_date = _business_days_before(STALE_BARS_MAX_WEEKDAYS + 1)
        model = _mock_model()

        with self.assertLogs("src.prediction.predict_unified", level="WARNING") as logs:
            result = _run(lambda m, s: _feature_frame(last_date), model)

        self.assertIsNone(result)
        model.predict.assert_not_called()
        joined = "\n".join(logs.output)
        self.assertIn("jp", joined)
        self.assertIn("5202", joined)
        self.assertIn(last_date.strftime("%Y-%m-%d"), joined)

    def test_holiday_gap_of_two_weekdays_still_predicts(self):
        last_date = _business_days_before(2)
        model = _mock_model()

        result = _run(lambda m, s: _feature_frame(last_date), model)

        self.assertIsNotNone(result)
        self.assertEqual(result.symbol, "5202")

    def test_gap_at_threshold_still_predicts(self):
        last_date = _business_days_before(STALE_BARS_MAX_WEEKDAYS)
        model = _mock_model()

        result = _run(lambda m, s: _feature_frame(last_date), model)

        self.assertIsNotNone(result)


class TestPredictAllUnifiedStaleCount(unittest.TestCase):
    def setUp(self):
        from src.prediction import predict_unified

        predict_unified._model_cache.clear()

    def test_completion_log_reports_stale_count(self):
        from contextlib import ExitStack

        from src.prediction.prediction_pipeline import predict_all_unified

        fresh = _latest_weekday()
        stale = _business_days_before(STALE_BARS_MAX_WEEKDAYS + 5)
        frames = {"5202": _feature_frame(stale), "7203": _feature_frame(fresh)}

        with ExitStack() as stack:
            for p in _patches(lambda m, s: frames[s], _mock_model()):
                stack.enter_context(p)
            stack.enter_context(
                patch(
                    "src.prediction.prediction_pipeline.get_active_symbols",
                    return_value=[("jp", "5202"), ("jp", "7203")],
                )
            )
            stack.enter_context(patch("src.prediction.predict_unified.preload_models"))
            with self.assertLogs("src.prediction.prediction_pipeline", level="INFO") as logs:
                rows = predict_all_unified(max_workers=2)

        self.assertEqual([r.symbol for r in rows], ["7203"])
        completion = [line for line in logs.output if "予測完了（統合モデル）" in line]
        self.assertEqual(len(completion), 1)
        self.assertIn("特徴量が古いため予想なし: 1銘柄", completion[0])


if __name__ == "__main__":
    unittest.main()
