"""ユニットテスト: domain.types"""

import unittest

import pandas as pd

from src.prediction.types import FeatureLoadResult, PredictionResult
from src.trading.types import TradingGateStatus


class TestFeatureLoadResult(unittest.TestCase):
    def test_success_status_and_attribute_access(self):
        result = FeatureLoadResult(
            status="success",
            market="jp",
            symbol="7203",
            X=pd.DataFrame({"a": [1, 2]}),
            y=pd.Series([0.1, 0.2]),
        )

        self.assertTrue(result.is_success)
        self.assertEqual(result.market, "jp")
        self.assertEqual(result.symbol, "7203")
        self.assertIsNone(result.reason)


class TestPredictionResult(unittest.TestCase):
    def test_to_dataframe_and_from_dataframe_roundtrip(self):
        result = PredictionResult(
            market="jp",
            symbol="7203",
            current_price=2500.0,
            avg_pred_price=2550.0,
            diff_ratio=0.02,
            model_count=2,
            confluence_score=80,
        )

        df = PredictionResult.to_dataframe([result])
        restored = PredictionResult.from_dataframe_row(df.iloc[0])

        self.assertEqual(restored.market, "jp")
        self.assertEqual(restored.symbol, "7203")
        self.assertEqual(restored.confluence_score, 80)
        self.assertAlmostEqual(restored.diff_ratio, 0.02)

    def test_to_dataframe_includes_multi_horizon_optional_fields(self):
        result = PredictionResult(
            market="us",
            symbol="AAPL",
            current_price=150.0,
            avg_pred_price=153.0,
            diff_ratio=0.02,
            model_count=3,
            avg_pred_price_3d=151.0,
            avg_pred_price_5d=152.0,
            avg_pred_price_10d=154.0,
            diff_ratio_3d=0.006,
            diff_ratio_5d=0.013,
            diff_ratio_10d=0.026,
        )

        df = PredictionResult.to_dataframe([result])

        self.assertIn("avg_pred_price_3d", df.columns)
        self.assertIn("diff_ratio_10d", df.columns)


class TestTradingGateStatus(unittest.TestCase):
    def test_defaults_are_safe(self):
        status = TradingGateStatus(is_allowed=True, stop_active=False)

        self.assertTrue(status.is_allowed)
        self.assertFalse(status.stop_active)
        self.assertEqual(status.daily_loss, 0.0)
        self.assertIsNone(status.daily_loss_limit)
        self.assertEqual(status.position_count, 0)


if __name__ == "__main__":
    unittest.main()
