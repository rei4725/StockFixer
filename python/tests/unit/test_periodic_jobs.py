"""ユニットテスト: src.orchestration.jobs.periodic の純粋ロジック部分"""

import unittest

import pandas as pd

from src.orchestration.jobs.periodic import _build_monthly_equity_series


def _series(n=5, value=100.0):
    idx = pd.bdate_range("2026-01-05", periods=n)
    return pd.Series(value, index=idx)


class TestBuildMonthlyEquitySeries(unittest.TestCase):
    def test_paper_trading_label_includes_start_date(self):
        result = _build_monthly_equity_series(_series(), pd.Series(dtype=float), None)
        self.assertIn("Paper Trading (since 2026-01-05)", result)
        self.assertEqual(len(result), 1)

    def test_allocation_bot_label_includes_its_own_start_date(self):
        allocation = pd.Series(50.0, index=pd.bdate_range("2026-03-02", periods=5))
        result = _build_monthly_equity_series(_series(), allocation, None)
        self.assertIn("Allocation Bot (since 2026-03-02)", result)

    def test_excludes_allocation_bot_when_empty(self):
        result = _build_monthly_equity_series(_series(), pd.Series(dtype=float), None)
        self.assertFalse(any(k.startswith("Allocation Bot") for k in result))

    def test_excludes_allocation_bot_when_fewer_than_5_points(self):
        """メインと同じ最低点数基準: 4点以下は運用開始直後のノイズとして除外"""
        allocation = pd.Series(50.0, index=pd.bdate_range("2026-09-01", periods=4))
        result = _build_monthly_equity_series(_series(), allocation, None)
        self.assertFalse(any(k.startswith("Allocation Bot") for k in result))

    def test_includes_benchmark_label_with_start_date_when_present(self):
        result = _build_monthly_equity_series(
            _series(), pd.Series(dtype=float), _series(value=5000.0)
        )
        self.assertIn("S&P 500 (since 2026-01-05)", result)


if __name__ == "__main__":
    unittest.main()
