import unittest
from unittest.mock import patch

import pandas as pd

from src.watchlist.ticker_list import get_nasdaq100_list, get_sp500_list


def _make_sp500_df():
    return pd.DataFrame(
        {"Symbol": ["AAPL", "BRK.B", "MSFT"], "Security": ["Apple", "Berkshire", "Microsoft"]}
    )


def _make_nasdaq100_df():
    return pd.DataFrame({"Ticker": ["AAPL", "MSFT"], "Company": ["Apple", "Microsoft"]})


class TestGetSp500List(unittest.TestCase):
    def test_returns_list_of_dicts(self):
        tables = [_make_sp500_df()]
        with patch("pandas.read_html", return_value=tables):
            result = get_sp500_list()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_each_entry_has_required_keys(self):
        tables = [_make_sp500_df()]
        with patch("pandas.read_html", return_value=tables):
            result = get_sp500_list()
        for item in result:
            self.assertIn("市場", item)
            self.assertIn("銘柄コード", item)
            self.assertIn("銘柄名", item)

    def test_market_is_us(self):
        tables = [_make_sp500_df()]
        with patch("pandas.read_html", return_value=tables):
            result = get_sp500_list()
        for item in result:
            self.assertEqual(item["市場"], "us")

    def test_dot_replaced_with_hyphen(self):
        tables = [_make_sp500_df()]
        with patch("pandas.read_html", return_value=tables):
            result = get_sp500_list()
        codes = [item["銘柄コード"] for item in result]
        self.assertIn("BRK-B", codes)
        self.assertNotIn("BRK.B", codes)

    def test_count_matches_dataframe(self):
        tables = [_make_sp500_df()]
        with patch("pandas.read_html", return_value=tables):
            result = get_sp500_list()
        self.assertEqual(len(result), 3)


class TestGetNasdaq100List(unittest.TestCase):
    def test_returns_list_of_dicts(self):
        tables = [None, None, None, None, _make_nasdaq100_df()]
        with patch("pandas.read_html", return_value=tables):
            result = get_nasdaq100_list()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_each_entry_has_required_keys(self):
        tables = [None, None, None, None, _make_nasdaq100_df()]
        with patch("pandas.read_html", return_value=tables):
            result = get_nasdaq100_list()
        for item in result:
            self.assertIn("市場", item)
            self.assertIn("銘柄コード", item)
            self.assertIn("銘柄名", item)

    def test_market_is_us(self):
        tables = [None, None, None, None, _make_nasdaq100_df()]
        with patch("pandas.read_html", return_value=tables):
            result = get_nasdaq100_list()
        for item in result:
            self.assertEqual(item["市場"], "us")
