"""ユニットテスト: prediction/drift_monitor (R-274)"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.prediction.drift_monitor import (
    DriftMonitorResult,
    _load_weekly_hit_rates_direct,
    _load_weekly_hit_rates_from_snapshots,
    check_weekly_hit_rate_drift,
)


class TestLoadWeeklyHitRatesFromSnapshots(unittest.TestCase):
    @patch("src.prediction.drift_monitor.load_weekly_accuracy_snapshots")
    def test_returns_empty_when_no_snapshots(self, mock_load):
        mock_load.return_value = pd.DataFrame()
        result = _load_weekly_hit_rates_from_snapshots(n_weeks=4)
        self.assertTrue(result.empty)

    @patch("src.prediction.drift_monitor.load_weekly_accuracy_snapshots")
    def test_aggregates_by_week(self, mock_load):
        mock_load.return_value = pd.DataFrame(
            {
                "week_start": ["2026-05-04", "2026-05-04", "2026-04-27", "2026-04-27"],
                "market": ["jp", "us", "jp", "us"],
                "symbol": ["7203", "AAPL", "7203", "AAPL"],
                "direction_accuracy": [0.6, 0.8, 0.7, 0.5],
                "mean_abs_error": [0.01, 0.02, 0.015, 0.025],
                "n_samples": [10, 10, 10, 10],
            }
        )
        result = _load_weekly_hit_rates_from_snapshots(n_weeks=4)
        self.assertEqual(len(result), 2)
        self.assertIn("week_start", result.columns)
        self.assertIn("hit_rate", result.columns)
        # 2026-05-04 週の平均: (0.6 + 0.8) / 2 = 0.7
        week_05_04 = result[result["week_start"] == "2026-05-04"].iloc[0]
        self.assertAlmostEqual(week_05_04["hit_rate"], 0.7)

    @patch("src.prediction.drift_monitor.load_weekly_accuracy_snapshots")
    def test_sorted_descending(self, mock_load):
        mock_load.return_value = pd.DataFrame(
            {
                "week_start": ["2026-04-27", "2026-05-04"],
                "market": ["jp", "jp"],
                "symbol": ["7203", "7203"],
                "direction_accuracy": [0.6, 0.7],
                "mean_abs_error": [0.01, 0.01],
                "n_samples": [10, 10],
            }
        )
        result = _load_weekly_hit_rates_from_snapshots(n_weeks=4)
        self.assertEqual(result.iloc[0]["week_start"], "2026-05-04")


class TestCheckWeeklyHitRateDrift(unittest.TestCase):
    def _make_weekly_df(self, weeks_data: list[tuple[str, float]]) -> pd.DataFrame:
        return pd.DataFrame(
            {"week_start": [w for w, _ in weeks_data], "hit_rate": [h for _, h in weeks_data]}
        )

    @patch("src.prediction.drift_monitor._load_weekly_hit_rates")
    def test_returns_not_drifted_when_data_insufficient(self, mock_load):
        mock_load.return_value = pd.DataFrame()
        result = check_weekly_hit_rate_drift(weeks=4, threshold=0.05)
        self.assertIsInstance(result, DriftMonitorResult)
        self.assertFalse(result.is_drifted)
        self.assertIsNone(result.current_hit_rate)

    @patch("src.prediction.drift_monitor._load_weekly_hit_rates")
    def test_detects_drift_when_below_threshold(self, mock_load):
        # 当週 0.50, 過去4週平均 0.65 → 低下率 ≈ 23% > 5% → ドリフト
        mock_load.return_value = self._make_weekly_df(
            [
                ("2026-05-11", 0.50),
                ("2026-05-04", 0.65),
                ("2026-04-27", 0.63),
                ("2026-04-20", 0.67),
                ("2026-04-13", 0.65),
            ]
        )
        result = check_weekly_hit_rate_drift(weeks=4, threshold=0.05)
        self.assertTrue(result.is_drifted)
        self.assertAlmostEqual(result.current_hit_rate, 0.50)
        self.assertAlmostEqual(result.avg_hit_rate, 0.65)
        self.assertGreater(result.drop_ratio, 0.05)

    @patch("src.prediction.drift_monitor._load_weekly_hit_rates")
    def test_no_drift_when_within_threshold(self, mock_load):
        # 当週 0.63, 過去4週平均 0.65 → 低下率 ≈ 3% < 5% → ドリフトなし
        mock_load.return_value = self._make_weekly_df(
            [
                ("2026-05-11", 0.63),
                ("2026-05-04", 0.65),
                ("2026-04-27", 0.64),
                ("2026-04-20", 0.66),
                ("2026-04-13", 0.65),
            ]
        )
        result = check_weekly_hit_rate_drift(weeks=4, threshold=0.05)
        self.assertFalse(result.is_drifted)

    @patch("src.prediction.drift_monitor._load_weekly_hit_rates")
    def test_drop_ratio_calculation(self, mock_load):
        # avg=0.80, current=0.72 → drop = (0.80-0.72)/0.80 = 0.10
        mock_load.return_value = self._make_weekly_df(
            [
                ("2026-05-11", 0.72),
                ("2026-05-04", 0.80),
                ("2026-04-27", 0.80),
            ]
        )
        result = check_weekly_hit_rate_drift(weeks=2, threshold=0.05)
        self.assertAlmostEqual(result.drop_ratio, 0.10, places=3)
        self.assertTrue(result.is_drifted)

    @patch("src.prediction.drift_monitor._load_weekly_hit_rates")
    def test_result_fields_populated(self, mock_load):
        mock_load.return_value = self._make_weekly_df(
            [
                ("2026-05-11", 0.60),
                ("2026-05-04", 0.70),
                ("2026-04-27", 0.70),
            ]
        )
        result = check_weekly_hit_rate_drift(weeks=2, threshold=0.05)
        self.assertEqual(result.current_week, "2026-05-11")
        self.assertEqual(result.alert_weeks, 2)
        self.assertAlmostEqual(result.alert_threshold, 0.05)
        self.assertIsNotNone(result.checked_at)

    @patch("src.prediction.drift_monitor._load_weekly_hit_rates")
    def test_single_week_data_returns_no_drift(self, mock_load):
        mock_load.return_value = self._make_weekly_df([("2026-05-11", 0.50)])
        result = check_weekly_hit_rate_drift(weeks=4, threshold=0.05)
        self.assertFalse(result.is_drifted)
        self.assertIsNone(result.current_hit_rate)

    @patch("src.prediction.drift_monitor._load_weekly_hit_rates")
    def test_improvement_not_flagged(self, mock_load):
        # 当週が平均より高い場合 (改善) はドリフトなし
        mock_load.return_value = self._make_weekly_df(
            [
                ("2026-05-11", 0.80),
                ("2026-05-04", 0.60),
                ("2026-04-27", 0.60),
            ]
        )
        result = check_weekly_hit_rate_drift(weeks=2, threshold=0.05)
        self.assertFalse(result.is_drifted)
        self.assertLessEqual(result.drop_ratio, 0.0)


class TestLoadWeeklyHitRatesDirect(unittest.TestCase):
    @patch("src.prediction.drift_monitor._db_connection")
    def test_returns_dataframe_on_success(self, mock_ctx):
        mock_con = mock_ctx.return_value.__enter__.return_value
        mock_con.execute.return_value.fetchdf.return_value = pd.DataFrame(
            {"week_start": ["2026-05-04", "2026-04-27"], "hit_rate": [0.65, 0.70]}
        )
        result = _load_weekly_hit_rates_direct(n_weeks=4, horizon=1)
        self.assertFalse(result.empty)
        self.assertIn("week_start", result.columns)
        self.assertIn("hit_rate", result.columns)

    @patch("src.prediction.drift_monitor._db_connection")
    def test_returns_empty_on_empty_query(self, mock_ctx):
        mock_con = mock_ctx.return_value.__enter__.return_value
        mock_con.execute.return_value.fetchdf.return_value = pd.DataFrame()
        result = _load_weekly_hit_rates_direct(n_weeks=4, horizon=1)
        self.assertTrue(result.empty)

    @patch("src.prediction.drift_monitor._db_connection")
    def test_returns_empty_on_exception(self, mock_ctx):
        mock_con = mock_ctx.return_value.__enter__.return_value
        mock_con.execute.side_effect = Exception("DB error")
        result = _load_weekly_hit_rates_direct(n_weeks=4, horizon=1)
        self.assertTrue(result.empty)


class TestCheckWeeklyHitRateDriftEdgeCases(unittest.TestCase):
    @patch("src.prediction.drift_monitor._load_weekly_hit_rates")
    def test_zero_avg_hit_rate_returns_zero_drop_ratio(self, mock_load):
        """avg_hit_rate が 0 の場合 drop_ratio は 0 でドリフトなし"""
        mock_load.return_value = pd.DataFrame(
            {
                "week_start": ["2026-05-11", "2026-05-04"],
                "hit_rate": [0.0, 0.0],
            }
        )
        result = check_weekly_hit_rate_drift(weeks=1, threshold=0.05)
        self.assertAlmostEqual(result.drop_ratio, 0.0)
        self.assertFalse(result.is_drifted)


if __name__ == "__main__":
    unittest.main()
