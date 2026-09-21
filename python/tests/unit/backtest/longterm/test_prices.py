"""longterm/prices.py の価格ヘルパー。"""

import unittest

import pandas as pd

from src.backtest.longterm import prices


def _frame(dates, closes):
    return pd.DataFrame({"date": dates, "Close": closes})


class TestBuildCalendar(unittest.TestCase):
    def test_union_of_trading_days_within_window(self):
        price_map = {
            "A": _frame(["2024-01-02", "2024-01-03"], [10.0, 11.0]),
            "B": _frame(["2024-01-03", "2024-01-04"], [20.0, 21.0]),
        }
        cal = prices.build_calendar(price_map, "2024-01-01", "2024-01-03")
        self.assertEqual(cal, ["2024-01-02", "2024-01-03"])


class TestMakeRescreenDates(unittest.TestCase):
    def test_quarterly_rounds_forward_to_trading_day(self):
        cal = ["2024-01-02", "2024-04-02", "2024-07-02"]
        out = prices.make_rescreen_dates(cal, "2024-01-01", "quarterly")
        self.assertEqual(out, ["2024-01-02", "2024-04-02", "2024-07-02"])

    def test_unknown_freq_falls_back_to_quarterly(self):
        cal = ["2024-01-02", "2024-04-02"]
        self.assertEqual(
            prices.make_rescreen_dates(cal, "2024-01-01", "bogus"),
            prices.make_rescreen_dates(cal, "2024-01-01", "quarterly"),
        )

    def test_empty_calendar_returns_empty(self):
        self.assertEqual(prices.make_rescreen_dates([], "2024-01-01", "quarterly"), [])


class TestCloseLookups(unittest.TestCase):
    def test_forward_fills_missing_days(self):
        price_map = {"A": _frame(["2024-01-02", "2024-01-04"], [10.0, 12.0])}
        cal = ["2024-01-02", "2024-01-03", "2024-01-04"]
        out = prices.close_lookups(price_map, cal)
        self.assertEqual(out["A"]["2024-01-03"], 10.0)
        self.assertEqual(out["A"]["2024-01-04"], 12.0)


if __name__ == "__main__":
    unittest.main()
