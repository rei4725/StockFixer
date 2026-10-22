import unittest
from unittest.mock import patch

from src.utils.sector_constraints import filter_by_sector_cap, get_symbol_sector


class TestFilterBySectorCap(unittest.TestCase):
    def _sector(self, item: str) -> str:
        return {"AAPL": "Technology", "MSFT": "Technology", "JPM": "Financials"}.get(
            item, "Unknown"
        )

    def test_no_limit_returns_all_items(self):
        items = ["AAPL", "MSFT", "JPM"]
        result = filter_by_sector_cap(items, max_sector_positions=0, sector_getter=self._sector)
        self.assertEqual(result, items)

    def test_negative_limit_returns_all_items(self):
        items = ["AAPL", "MSFT", "JPM"]
        result = filter_by_sector_cap(items, max_sector_positions=-1, sector_getter=self._sector)
        self.assertEqual(result, items)

    def test_cap_limits_per_sector(self):
        items = ["AAPL", "MSFT", "JPM"]
        result = filter_by_sector_cap(items, max_sector_positions=1, sector_getter=self._sector)
        self.assertIn("AAPL", result)
        self.assertNotIn("MSFT", result)
        self.assertIn("JPM", result)

    def test_cap_of_two_allows_two_per_sector(self):
        items = ["AAPL", "MSFT", "JPM"]
        result = filter_by_sector_cap(items, max_sector_positions=2, sector_getter=self._sector)
        self.assertEqual(len(result), 3)

    def test_empty_input_returns_empty(self):
        result = filter_by_sector_cap([], max_sector_positions=1, sector_getter=self._sector)
        self.assertEqual(result, [])

    def test_preserves_order(self):
        items = ["AAPL", "JPM", "MSFT"]
        result = filter_by_sector_cap(items, max_sector_positions=1, sector_getter=self._sector)
        self.assertEqual(result.index("AAPL"), 0)
        self.assertEqual(result.index("JPM"), 1)


class TestGetSymbolSector(unittest.TestCase):
    def test_returns_sector_from_yfinance(self):
        mock_ticker = unittest.mock.MagicMock()
        mock_ticker.info = {"sector": "Technology"}
        with patch("src.utils.sector_constraints.yf.Ticker", return_value=mock_ticker):
            result = get_symbol_sector.cache_clear() or get_symbol_sector("us", "AAPL")
        self.assertIn("Technology", result)

    def test_falls_back_to_industry_when_no_sector(self):
        get_symbol_sector.cache_clear()
        mock_ticker = unittest.mock.MagicMock()
        mock_ticker.info = {"industry": "Software"}
        with patch("src.utils.sector_constraints.yf.Ticker", return_value=mock_ticker):
            result = get_symbol_sector("us", "TEST")
        self.assertIn("Software", result)

    def test_returns_unknown_on_exception(self):
        get_symbol_sector.cache_clear()
        with patch("src.utils.sector_constraints.yf.Ticker", side_effect=RuntimeError("network")):
            result = get_symbol_sector("us", "ERR")
        self.assertIn("UNKNOWN", result)

    def test_returns_unknown_when_sector_empty(self):
        get_symbol_sector.cache_clear()
        mock_ticker = unittest.mock.MagicMock()
        mock_ticker.info = {}
        with patch("src.utils.sector_constraints.yf.Ticker", return_value=mock_ticker):
            result = get_symbol_sector("us", "EMPTY")
        self.assertIn("UNKNOWN", result)

    def test_exception_is_not_cached_and_retried(self):
        get_symbol_sector.cache_clear()
        mock_ticker = unittest.mock.MagicMock()
        mock_ticker.info = {"sector": "Technology"}
        with patch(
            "src.utils.sector_constraints.yf.Ticker",
            side_effect=[RuntimeError("429"), mock_ticker],
        ) as mock_cls:
            first = get_symbol_sector("us", "RETRY")
            second = get_symbol_sector("us", "RETRY")
        self.assertIn("UNKNOWN", first)
        self.assertEqual(second, "Technology")
        self.assertEqual(mock_cls.call_count, 2)

    def test_empty_sector_is_not_cached_and_retried(self):
        get_symbol_sector.cache_clear()
        empty_ticker = unittest.mock.MagicMock()
        empty_ticker.info = {}
        good_ticker = unittest.mock.MagicMock()
        good_ticker.info = {"sector": "Financials"}
        with patch(
            "src.utils.sector_constraints.yf.Ticker",
            side_effect=[empty_ticker, good_ticker],
        ) as mock_cls:
            first = get_symbol_sector("us", "RETRY_EMPTY")
            second = get_symbol_sector("us", "RETRY_EMPTY")
        self.assertIn("UNKNOWN", first)
        self.assertEqual(second, "Financials")
        self.assertEqual(mock_cls.call_count, 2)

    def test_resolved_sector_is_cached(self):
        get_symbol_sector.cache_clear()
        mock_ticker = unittest.mock.MagicMock()
        mock_ticker.info = {"sector": "Energy"}
        with patch("src.utils.sector_constraints.yf.Ticker", return_value=mock_ticker) as mock_cls:
            first = get_symbol_sector("us", "CACHED")
            second = get_symbol_sector("us", "CACHED")
        self.assertEqual(first, "Energy")
        self.assertEqual(second, "Energy")
        self.assertEqual(mock_cls.call_count, 1)

    def tearDown(self):
        get_symbol_sector.cache_clear()
