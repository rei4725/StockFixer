import unittest
from unittest.mock import patch

import pandas as pd

from config.settings import MAX_SECTOR_POSITIONS
from src.domain.sector_cap import filter_by_sector_cap
from src.trading.execution.selection import _apply_buy_sector_limit
from src.utils.sector_constraints import UNKNOWN_SECTOR, get_symbol_sector


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

    def test_initial_counts_are_included_in_cap(self):
        items = ["AAPL", "MSFT", "JPM"]
        result = filter_by_sector_cap(
            items,
            max_sector_positions=2,
            sector_getter=self._sector,
            initial_counts={"Technology": 1},
        )
        self.assertEqual(result, ["AAPL", "JPM"])


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

    def test_unresolved_symbols_share_one_sector(self):
        """#1469: 引けなかった銘柄は銘柄ごとの別の業種でなく、同じ1つの業種になる。"""
        get_symbol_sector.cache_clear()
        empty_ticker = unittest.mock.MagicMock()
        empty_ticker.info = {}
        with patch(
            "src.utils.sector_constraints.yf.Ticker",
            side_effect=[RuntimeError("429"), RuntimeError("429"), empty_ticker],
        ):
            sectors = {
                get_symbol_sector("jp", "1001"),
                get_symbol_sector("jp", "1002"),
                get_symbol_sector("us", "ETF"),
            }
        self.assertEqual(sectors, {UNKNOWN_SECTOR})

    def tearDown(self):
        get_symbol_sector.cache_clear()


class TestUnresolvedSectorCap(unittest.TestCase):
    """#1469: yfinance で業種が引けない日も同一業種の上限が効く。"""

    def setUp(self):
        get_symbol_sector.cache_clear()

    def tearDown(self):
        get_symbol_sector.cache_clear()

    def _candidates(self, symbols: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": s,
                    "diff_ratio": 0.05 - 0.001 * i,
                    "current_price": 1000.0,
                }
                for i, s in enumerate(symbols)
            ]
        )

    def test_lookup_failure_candidates_are_capped(self):
        symbols = [str(9000 + i) for i in range(MAX_SECTOR_POSITIONS + 2)]
        with patch("src.utils.sector_constraints.yf.Ticker", side_effect=RuntimeError("429")):
            limited = _apply_buy_sector_limit(
                self._candidates(symbols), max_sector_positions=MAX_SECTOR_POSITIONS
            )
        self.assertEqual(limited["symbol"].tolist(), symbols[:MAX_SECTOR_POSITIONS])

    def test_lookup_failure_holdings_count_toward_cap(self):
        symbols = [str(9000 + i) for i in range(MAX_SECTOR_POSITIONS + 2)]
        with patch("src.utils.sector_constraints.yf.Ticker", side_effect=RuntimeError("429")):
            limited = _apply_buy_sector_limit(
                self._candidates(symbols),
                max_sector_positions=MAX_SECTOR_POSITIONS,
                held_symbols={"8000"},
                market="jp",
            )
        self.assertEqual(limited["symbol"].tolist(), symbols[: MAX_SECTOR_POSITIONS - 1])

    def test_resolved_sectors_are_counted_separately(self):
        sectors = {"7200": "Auto", "7201": "Auto", "6758": "Technology"}

        def _ticker(ticker: str) -> unittest.mock.MagicMock:
            mock = unittest.mock.MagicMock()
            symbol = ticker.split(".")[0]
            if symbol in sectors:
                mock.info = {"sector": sectors[symbol]}
            else:
                type(mock).info = unittest.mock.PropertyMock(side_effect=RuntimeError("429"))
            return mock

        with patch("src.utils.sector_constraints.yf.Ticker", side_effect=_ticker):
            limited = _apply_buy_sector_limit(
                self._candidates(["7200", "7201", "9000", "9001", "6758"]),
                max_sector_positions=1,
            )
        self.assertEqual(limited["symbol"].tolist(), ["7200", "9000", "6758"])
        self.assertEqual(limited["sector"].tolist(), ["Auto", UNKNOWN_SECTOR, "Technology"])
