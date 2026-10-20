"""紙の売買の週の成績（results/paper/YYYY-Www.json）のユニットテスト（#812）"""

import json
import os
import tempfile
import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from src.trading.paper_equity import compute_weekly_results, write_paper_weekly_results

_INITIAL = 1_000_000.0
_COLUMNS = ["market", "symbol", "side", "qty", "fill_price", "filled_at"]


def _orders(rows):
    return pd.DataFrame(rows, columns=_COLUMNS)


def _price_loader(market, symbol):
    """前週金曜（2026-10-09）は 100 円、今週（2026-10-12〜16）は日ごとに上がり金曜 110 円。"""
    idx = pd.bdate_range("2026-09-28", "2026-10-16")
    prices = pd.Series(100.0, index=idx)
    prices[pd.Timestamp("2026-10-16")] = 110.0
    return prices


def _benchmark_fetcher(ticker, start, end):
    returns = {"1306.T": 0.02, "^GSPC": -0.01}
    return {"ticker": ticker, "total_return": returns[ticker], "start": start, "end": end}


class TestComputeWeeklyResults(unittest.TestCase):
    """決まった約定と終値から週の4つの値が出ること"""

    def setUp(self):
        # jp: 前週に 1,000 株を 100 円で買い、今週に 500 株を 100 円で買い増す（約定 1 件）
        self.orders = _orders(
            [
                ("jp", "7203", 1, 1000, 100.0, pd.Timestamp("2026-10-07")),
                ("jp", "7203", 1, 500, 100.0, pd.Timestamp("2026-10-13")),
            ]
        )

    def test_values_per_market(self):
        result = compute_weekly_results(
            self.orders, _price_loader, _INITIAL, date(2026, 10, 17), _benchmark_fetcher
        )

        self.assertEqual(result["week"], "2026-W42")
        jp = result["markets"]["jp"]
        # 前週末の評価額 1,000,000 → 今週末 1,000,000 + 1,500 株 × 10 円 = 1,015,000
        self.assertAlmostEqual(jp["weekly_return"], 0.015)
        self.assertAlmostEqual(jp["benchmark_return"], 0.02)
        self.assertAlmostEqual(jp["excess_return"], -0.005)
        self.assertEqual(jp["fills"], 1)
        self.assertEqual(jp["benchmark_ticker"], "1306.T")

        us = result["markets"]["us"]
        self.assertAlmostEqual(us["weekly_return"], 0.0)
        self.assertAlmostEqual(us["benchmark_return"], -0.01)
        self.assertAlmostEqual(us["excess_return"], 0.01)
        self.assertEqual(us["fills"], 0)
        self.assertEqual(us["benchmark_ticker"], "^GSPC")

    def test_benchmark_period_is_previous_week_end_to_this_week_end(self):
        calls = []

        def fetcher(ticker, start, end):
            calls.append((ticker, start, end))
            return {"ticker": ticker, "total_return": 0.0, "start": start, "end": end}

        compute_weekly_results(self.orders, _price_loader, _INITIAL, date(2026, 10, 17), fetcher)

        # yfinance の end は含まないため、金曜の翌日を渡す
        self.assertIn(("1306.T", "2026-10-09", "2026-10-17"), calls)

    def test_benchmark_unavailable_gives_null(self):
        def failing(ticker, start, end):
            return {"ticker": ticker, "total_return": None, "start": start, "end": end}

        result = compute_weekly_results(
            self.orders, _price_loader, _INITIAL, date(2026, 10, 17), failing
        )
        jp = result["markets"]["jp"]
        self.assertIsNone(jp["benchmark_return"])
        self.assertIsNone(jp["excess_return"])
        self.assertAlmostEqual(jp["weekly_return"], 0.015)

    def test_benchmark_exception_gives_null(self):
        def raising(ticker, start, end):
            raise RuntimeError("network")

        result = compute_weekly_results(
            self.orders, _price_loader, _INITIAL, date(2026, 10, 17), raising
        )
        self.assertIsNone(result["markets"]["jp"]["benchmark_return"])
        self.assertIsNone(result["markets"]["jp"]["excess_return"])

    def test_first_week_uses_initial_balance_as_base(self):
        """前週末より前に約定が無い週は、元手を基準にすること"""
        orders = _orders([("jp", "7203", 1, 1000, 100.0, pd.Timestamp("2026-10-13"))])
        result = compute_weekly_results(
            orders, _price_loader, _INITIAL, date(2026, 10, 17), _benchmark_fetcher
        )
        # 1,000 株 × 10 円 = +10,000
        self.assertAlmostEqual(result["markets"]["jp"]["weekly_return"], 0.01)


class TestWritePaperWeeklyResults(unittest.TestCase):
    """週の道からファイルに書くこと"""

    @patch("src.trading.paper_equity._default_price_loader", side_effect=_price_loader)
    @patch("src.trading.paper_equity._load_filled_orders")
    def test_writes_iso_week_json(self, mock_load, _mock_prices):
        mock_load.return_value = _orders(
            [("jp", "7203", 1, 1000, 100.0, pd.Timestamp("2026-10-07"))]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_paper_weekly_results(date(2026, 10, 17), _benchmark_fetcher, tmp)
            self.assertEqual(path, os.path.join(tmp, "2026-W42.json"))
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        self.assertAlmostEqual(data["markets"]["jp"]["weekly_return"], 0.01)
        self.assertEqual(data["markets"]["jp"]["fills"], 0)

    @patch("src.trading.paper_equity._load_filled_orders")
    def test_no_orders_writes_nothing(self, mock_load):
        mock_load.return_value = _orders([])
        with tempfile.TemporaryDirectory() as tmp:
            path = write_paper_weekly_results(date(2026, 10, 17), _benchmark_fetcher, tmp)
            self.assertIsNone(path)
            self.assertEqual(os.listdir(tmp), [])


class TestWeeklyReportCallsPaperResults(unittest.TestCase):
    """週のレポートの道が週の成績を書き、失敗しても止まらないこと"""

    @patch("src.trading.paper_equity.write_paper_weekly_results", side_effect=RuntimeError("x"))
    def test_failure_does_not_propagate(self, mock_write):
        from src.orchestration.jobs.weekly import run_weekly_paper_results

        run_weekly_paper_results()  # 例外が外に出ないこと
        mock_write.assert_called_once()


if __name__ == "__main__":
    unittest.main()
