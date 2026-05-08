"""データ品質チェックモジュールのユニットテスト"""

import glob
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.utils.data_path_utils as path_utils
import src.utils.db as db_module
from src.market_data.quality_check import (
    QualityCheckResult,
    QualityIssue,
    _check_consecutive_missing,
    _check_nan_rate,
    _check_zero_volume,
    run_quality_checks,
)


def _make_ohlcv(periods=60, nan_cols=None, zero_vol_start=None, zero_vol_days=0):
    """テスト用 OHLCV DataFrame を生成する。"""
    dates = pd.date_range("2023-01-01", periods=periods, freq="D")
    df = pd.DataFrame(
        {
            "Open": np.linspace(100, 120, periods),
            "High": np.linspace(101, 121, periods),
            "Low": np.linspace(99, 119, periods),
            "Close": np.linspace(100, 120, periods),
            "Volume": np.full(periods, 1000, dtype=float),
        },
        index=dates,
    )
    if nan_cols:
        for col, start, length in nan_cols:
            df.iloc[start : start + length, df.columns.get_loc(col)] = np.nan
    if zero_vol_start is not None:
        df.iloc[zero_vol_start : zero_vol_start + zero_vol_days, df.columns.get_loc("Volume")] = 0
    return df


class TestCheckNanRate(unittest.TestCase):
    def test_no_nan_returns_empty(self):
        df = _make_ohlcv(60)
        issues = _check_nan_rate(df, threshold=0.05)
        self.assertEqual(issues, [])

    def test_nan_rate_above_threshold_returns_warning(self):
        df = _make_ohlcv(60, nan_cols=[("Close", 0, 10)])  # 10/60 ≈ 16.7%
        issues = _check_nan_rate(df, threshold=0.05)
        self.assertTrue(any(i.check == "nan_rate" and i.level == "warning" for i in issues))

    def test_nan_rate_below_threshold_returns_empty(self):
        df = _make_ohlcv(100, nan_cols=[("Close", 0, 4)])  # 4/100 = 4% < 5%
        issues = _check_nan_rate(df, threshold=0.05)
        col_issues = [i for i in issues if "Close" in i.detail]
        self.assertEqual(col_issues, [])

    def test_detail_contains_column_name(self):
        df = _make_ohlcv(60, nan_cols=[("Open", 0, 10)])
        issues = _check_nan_rate(df, threshold=0.05)
        self.assertTrue(any("Open" in i.detail for i in issues))


class TestCheckConsecutiveMissing(unittest.TestCase):
    def test_no_missing_returns_empty(self):
        df = _make_ohlcv(60)
        issues = _check_consecutive_missing(df, n_days=5)
        self.assertEqual(issues, [])

    def test_consecutive_missing_above_threshold_returns_error(self):
        df = _make_ohlcv(60, nan_cols=[("Close", 10, 7)])  # 7日連続
        issues = _check_consecutive_missing(df, n_days=5)
        self.assertTrue(any(i.check == "consecutive_missing" and i.level == "error" for i in issues))

    def test_consecutive_missing_below_threshold_returns_empty(self):
        df = _make_ohlcv(60, nan_cols=[("Close", 10, 4)])  # 4日連続 < 閾値5
        issues = _check_consecutive_missing(df, n_days=5)
        self.assertEqual(issues, [])

    def test_no_close_column_returns_empty(self):
        df = pd.DataFrame({"Open": [1, 2, 3], "Volume": [100, 200, 300]})
        issues = _check_consecutive_missing(df, n_days=2)
        self.assertEqual(issues, [])

    def test_exactly_at_threshold_returns_error(self):
        df = _make_ohlcv(60, nan_cols=[("Close", 10, 5)])  # ちょうど5日 = 閾値5
        issues = _check_consecutive_missing(df, n_days=5)
        self.assertTrue(any(i.check == "consecutive_missing" for i in issues))


class TestCheckZeroVolume(unittest.TestCase):
    def test_no_zero_volume_returns_empty(self):
        df = _make_ohlcv(60)
        issues = _check_zero_volume(df, n_days=5)
        self.assertEqual(issues, [])

    def test_consecutive_zero_volume_above_threshold_returns_warning(self):
        df = _make_ohlcv(60, zero_vol_start=10, zero_vol_days=7)
        issues = _check_zero_volume(df, n_days=5)
        self.assertTrue(any(i.check == "zero_volume" and i.level == "warning" for i in issues))

    def test_consecutive_zero_volume_below_threshold_returns_empty(self):
        df = _make_ohlcv(60, zero_vol_start=10, zero_vol_days=4)
        issues = _check_zero_volume(df, n_days=5)
        self.assertEqual(issues, [])

    def test_no_volume_column_returns_empty(self):
        df = pd.DataFrame({"Close": [100, 101, 102, 103, 104, 105]})
        issues = _check_zero_volume(df, n_days=3)
        self.assertEqual(issues, [])

    def test_detail_mentions_delisting(self):
        df = _make_ohlcv(60, zero_vol_start=10, zero_vol_days=7)
        issues = _check_zero_volume(df, n_days=5)
        self.assertTrue(any("上場廃止" in i.detail for i in issues))


class TestRunQualityChecks(unittest.TestCase):
    def test_clean_data_passes(self):
        df = _make_ohlcv(60)
        result = run_quality_checks("us", "AAPL", df)
        self.assertIsInstance(result, QualityCheckResult)
        self.assertTrue(result.passed)
        self.assertEqual(result.issues, [])
        self.assertEqual(result.market, "us")
        self.assertEqual(result.symbol, "AAPL")

    def test_error_issue_marks_passed_false(self):
        df = _make_ohlcv(60, nan_cols=[("Close", 10, 7)])
        result = run_quality_checks("us", "BAD", df, consecutive_missing_days=5)
        self.assertFalse(result.passed)
        self.assertTrue(any(i.level == "error" for i in result.issues))

    def test_warning_only_keeps_passed_true(self):
        # NaN率超過（warning）だけで error なし
        df = _make_ohlcv(60, nan_cols=[("Open", 0, 10)])
        result = run_quality_checks("us", "WARN", df, nan_threshold=0.05)
        self.assertTrue(result.passed)
        self.assertTrue(any(i.level == "warning" for i in result.issues))

    def test_zero_volume_generates_warning(self):
        df = _make_ohlcv(60, zero_vol_start=5, zero_vol_days=8)
        result = run_quality_checks("us", "ZV", df, zero_volume_days=5)
        self.assertTrue(any(i.check == "zero_volume" for i in result.issues))


class TestInsertQualityLog(unittest.TestCase):
    """insert_quality_log の DB 書き込みテスト"""

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

    def test_insert_and_read_back(self):
        from src.utils.db._connection import _db_connection
        from src.utils.db.quality_log import insert_quality_log

        issues = [
            {"check": "nan_rate", "level": "warning", "detail": "テスト警告"},
            {"check": "consecutive_missing", "level": "error", "detail": "テストエラー"},
        ]
        insert_quality_log("us", "AAPL", issues)

        with _db_connection() as con:
            rows = con.execute(
                "SELECT market, symbol, check_name, level FROM data_quality_log"
            ).fetchall()

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], "us")
        self.assertEqual(rows[0][1], "AAPL")
        levels = {r[3] for r in rows}
        self.assertIn("warning", levels)
        self.assertIn("error", levels)

    def test_empty_issues_writes_nothing(self):
        from src.utils.db._connection import _db_connection
        from src.utils.db.quality_log import insert_quality_log

        insert_quality_log("us", "EMPTY", [])

        with _db_connection() as con:
            count = con.execute("SELECT COUNT(*) FROM data_quality_log").fetchone()[0]
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
