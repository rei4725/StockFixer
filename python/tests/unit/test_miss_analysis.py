"""miss_analysis モジュールのユニットテスト"""

import glob
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd

import src.utils.data_path_utils as path_utils
import src.utils.db as db_module
from src.prediction.db import load_top_prediction_misses, save_prediction_accuracy
from src.prediction.miss_analysis import (
    MissCauseFeature,
    _extract_dates,
    _warn_repeated_features,
    analyze_miss_causes,
    run_miss_analysis_batch,
)


def _recent_predicted_at(days_ago: int = 1) -> str:
    """since_days=30 フィルターに確実に収まる最近の predicted_at 文字列を返す。"""
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y%m%d_120000")


class _TmpDbTestCase(unittest.TestCase):
    def setUp(self):
        db_module.close_connection()
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_db = os.path.join(self.tmp_dir, "test.duckdb")
        self._orig_get_db_path = path_utils.get_db_path
        path_utils.get_db_path = lambda: self.tmp_db
        db_module.get_db_path = lambda: self.tmp_db
        db_module._tables_initialized = False

    def tearDown(self):
        db_module.close_connection()
        path_utils.get_db_path = self._orig_get_db_path
        db_module.get_db_path = self._orig_get_db_path
        for path in glob.glob(self.tmp_db + "*"):
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)


class TestExtractDates(unittest.TestCase):
    def test_standard_format(self):
        result = _extract_dates(["20260403_120000", "20260404_090000"])
        self.assertEqual(result, {"2026-04-03", "2026-04-04"})

    def test_empty_list(self):
        self.assertEqual(_extract_dates([]), set())

    def test_invalid_format_skipped(self):
        result = _extract_dates(["invalid", "20260403_120000"])
        self.assertEqual(result, {"2026-04-03"})

    def test_duplicate_dates_deduplicated(self):
        result = _extract_dates(["20260403_090000", "20260403_150000"])
        self.assertEqual(result, {"2026-04-03"})


class TestLoadTopPredictionMisses(_TmpDbTestCase):
    def _insert_accuracy(self, rows):
        save_prediction_accuracy(rows)

    def test_returns_empty_when_no_data(self):
        df = load_top_prediction_misses(horizon=1, top_n=5, since_days=30)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_returns_top_n_by_abs_error(self):
        recent = _recent_predicted_at()
        rows = [
            {
                "market": "us",
                "symbol": "AAPL",
                "model_name": "test",
                "predicted_at": recent,
                "horizon": 1,
                "predicted_ratio": 0.05,
                "actual_ratio": -0.03,
            },
            {
                "market": "us",
                "symbol": "GOOG",
                "model_name": "test",
                "predicted_at": recent,
                "horizon": 1,
                "predicted_ratio": 0.01,
                "actual_ratio": -0.00,
            },
        ]
        self._insert_accuracy(rows)
        df = load_top_prediction_misses(horizon=1, top_n=5, since_days=30)
        self.assertFalse(df.empty)
        self.assertIn("abs_error", df.columns)
        # AAPL の外れ幅 0.08 > GOOG の外れ幅 0.01
        self.assertEqual(df.iloc[0]["symbol"], "AAPL")
        self.assertAlmostEqual(df.iloc[0]["abs_error"], 0.08)

    def test_excludes_null_actual_ratio(self):
        rows = [
            {
                "market": "us",
                "symbol": "TSLA",
                "model_name": "test",
                "predicted_at": "20260501_120000",
                "horizon": 1,
                "predicted_ratio": 0.05,
                "actual_ratio": None,
            },
        ]
        self._insert_accuracy(rows)
        df = load_top_prediction_misses(horizon=1, top_n=5, since_days=30)
        self.assertTrue(df.empty)

    def test_top_n_limit(self):
        recent = _recent_predicted_at()
        rows = [
            {
                "market": "us",
                "symbol": f"SYM{i}",
                "model_name": "test",
                "predicted_at": recent,
                "horizon": 1,
                "predicted_ratio": 0.01 * i,
                "actual_ratio": -0.01 * i,
            }
            for i in range(1, 6)
        ]
        self._insert_accuracy(rows)
        df = load_top_prediction_misses(horizon=1, top_n=3, since_days=30)
        self.assertEqual(len(df), 3)


class TestAnalyzeMissCauses(unittest.TestCase):
    def _make_miss_df(self, market="us", symbol="AAPL"):
        return pd.DataFrame(
            [
                {
                    "market": market,
                    "symbol": symbol,
                    "model_name": "test",
                    "predicted_at": "20260401_120000",
                    "horizon": 1,
                    "predicted_ratio": 0.05,
                    "actual_ratio": -0.03,
                    "abs_error": 0.08,
                }
            ]
        )

    def _make_shap_df(self):
        return pd.DataFrame(
            [
                {"feature": "rsi_14", "shap_mean": 0.5, "shap_rank": 1, "trained_at": "20260301"},
                {
                    "feature": "volume_ma20",
                    "shap_mean": 0.3,
                    "shap_rank": 2,
                    "trained_at": "20260301",
                },
            ]
        )

    def _make_features_df(self, extreme=True):
        # 正常日は mean≈50, std≈2 になるよう安定した値を用意し、
        # 外れ日の値 (1.0) が確実に |z| > 1.5 になるようにする
        normal_dates = [f"2026-03-{d:02d}" for d in range(1, 21)]
        normal_rsi = [50.0 + (i % 5 - 2) * 0.5 for i in range(20)]  # 48.0〜51.0
        normal_vol = [1000.0 + (i % 5 - 2) * 10 for i in range(20)]  # 980〜1020
        miss_rsi = 1.0 if extreme else 50.0  # z ≈ -24 (extreme) or 0 (normal)
        miss_vol = 9000.0 if extreme else 1000.0
        base = pd.DataFrame(
            {
                "date": normal_dates + ["2026-04-01"],
                "rsi_14": normal_rsi + [miss_rsi],
                "volume_ma20": normal_vol + [miss_vol],
            }
        )
        return base

    def test_returns_empty_for_empty_miss_df(self):
        result = analyze_miss_causes("us", "AAPL", pd.DataFrame())
        self.assertEqual(result, [])

    def test_returns_empty_when_shap_missing(self):
        miss_df = self._make_miss_df()
        with patch("src.prediction.miss_analysis.load_shap_latest") as mock_shap, patch(
            "src.prediction.miss_analysis.load_stock_features"
        ) as mock_feat:
            mock_shap.return_value = pd.DataFrame()
            mock_feat.return_value = self._make_features_df()
            result = analyze_miss_causes("us", "AAPL", miss_df)
        self.assertEqual(result, [])

    def test_returns_empty_when_features_missing(self):
        miss_df = self._make_miss_df()
        with patch("src.prediction.miss_analysis.load_shap_latest") as mock_shap, patch(
            "src.prediction.miss_analysis.load_stock_features"
        ) as mock_feat:
            mock_shap.return_value = self._make_shap_df()
            mock_feat.return_value = None
            result = analyze_miss_causes("us", "AAPL", miss_df)
        self.assertEqual(result, [])

    def test_detects_extreme_feature_on_miss_day(self):
        miss_df = self._make_miss_df()
        with patch("src.prediction.miss_analysis.load_shap_latest") as mock_shap, patch(
            "src.prediction.miss_analysis.load_stock_features"
        ) as mock_feat:
            mock_shap.return_value = self._make_shap_df()
            mock_feat.return_value = self._make_features_df(extreme=True)
            result = analyze_miss_causes("us", "AAPL", miss_df)
        features_found = [r.feature for r in result]
        self.assertIn("rsi_14", features_found)

    def test_no_causes_when_no_extreme_values(self):
        miss_df = self._make_miss_df()
        with patch("src.prediction.miss_analysis.load_shap_latest") as mock_shap, patch(
            "src.prediction.miss_analysis.load_stock_features"
        ) as mock_feat:
            mock_shap.return_value = self._make_shap_df()
            mock_feat.return_value = self._make_features_df(extreme=False)
            result = analyze_miss_causes("us", "AAPL", miss_df)
        self.assertEqual(result, [])

    def test_result_sorted_by_shap_rank(self):
        miss_df = self._make_miss_df()
        with patch("src.prediction.miss_analysis.load_shap_latest") as mock_shap, patch(
            "src.prediction.miss_analysis.load_stock_features"
        ) as mock_feat:
            mock_shap.return_value = self._make_shap_df()
            mock_feat.return_value = self._make_features_df(extreme=True)
            result = analyze_miss_causes("us", "AAPL", miss_df)
        if len(result) > 1:
            for i in range(len(result) - 1):
                self.assertLessEqual(result[i].shap_rank, result[i + 1].shap_rank)

    def test_warn_repeated_features_logs_warning(self):
        causes = {
            ("us", "AAPL"): [MissCauseFeature("rsi_14", 1, 0.5, 3)],
            ("us", "GOOG"): [MissCauseFeature("rsi_14", 2, 0.4, 2)],
            ("jp", "7203"): [MissCauseFeature("rsi_14", 1, 0.6, 1)],
        }
        with self.assertLogs("src.prediction.miss_analysis", level="WARNING") as cm:
            _warn_repeated_features(causes, threshold=3)
        self.assertTrue(any("rsi_14" in line for line in cm.output))


class TestRunMissAnalysisBatch(unittest.TestCase):
    def test_empty_miss_df_returns_empty(self):
        result = run_miss_analysis_batch(pd.DataFrame())
        self.assertEqual(result, {})

    def test_groups_by_market_symbol(self):
        miss_df = pd.DataFrame(
            [
                {
                    "market": "us",
                    "symbol": "AAPL",
                    "model_name": "test",
                    "predicted_at": "20260401_120000",
                    "horizon": 1,
                    "predicted_ratio": 0.05,
                    "actual_ratio": -0.03,
                    "abs_error": 0.08,
                },
                {
                    "market": "us",
                    "symbol": "GOOG",
                    "model_name": "test",
                    "predicted_at": "20260401_120000",
                    "horizon": 1,
                    "predicted_ratio": 0.04,
                    "actual_ratio": -0.02,
                    "abs_error": 0.06,
                },
            ]
        )
        with patch("src.prediction.miss_analysis.analyze_miss_causes") as mock_analyze:
            mock_analyze.return_value = [MissCauseFeature("rsi_14", 1, 0.5, 2)]
            result = run_miss_analysis_batch(miss_df)
        self.assertEqual(mock_analyze.call_count, 2)
        self.assertIn(("us", "AAPL"), result)
        self.assertIn(("us", "GOOG"), result)


if __name__ == "__main__":
    unittest.main()
