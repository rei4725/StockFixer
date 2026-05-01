import os
import tempfile
import unittest
from datetime import date

import src.utils.data_path_utils as path_utils
import src.utils.db as db_module
from src.utils.db.index_membership import (
    _normalize_symbol,
    load_index_membership_symbols_as_of,
    save_index_membership_snapshot,
)


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


class TestNormalizeSymbol(unittest.TestCase):
    def test_us_market_uppercases(self):
        self.assertEqual(_normalize_symbol("aapl", "us"), "AAPL")

    def test_jp_market_preserves_case(self):
        self.assertEqual(_normalize_symbol("7203", "jp"), "7203")

    def test_strips_whitespace(self):
        self.assertEqual(_normalize_symbol("  AAPL  ", "us"), "AAPL")

    def test_empty_string_returns_empty(self):
        self.assertEqual(_normalize_symbol("", "us"), "")

    def test_whitespace_only_returns_empty(self):
        self.assertEqual(_normalize_symbol("   ", "us"), "")


class TestSaveIndexMembershipSnapshot(_TmpDbTestCase):
    def test_returns_zero_for_empty_symbols(self):
        result = save_index_membership_snapshot("us", [])
        self.assertEqual(result, 0)

    def test_returns_count_of_saved_rows(self):
        result = save_index_membership_snapshot("us", ["AAPL", "GOOG", "MSFT"])
        self.assertEqual(result, 3)

    def test_deduplicates_symbols(self):
        result = save_index_membership_snapshot("us", ["AAPL", "AAPL", "GOOG"])
        self.assertEqual(result, 2)

    def test_filters_blank_symbols(self):
        result = save_index_membership_snapshot("us", ["AAPL", "", "  ", "GOOG"])
        self.assertEqual(result, 2)

    def test_returns_zero_when_all_blank(self):
        result = save_index_membership_snapshot("us", ["", "  "])
        self.assertEqual(result, 0)

    def test_uses_provided_snapshot_date(self):
        save_index_membership_snapshot("us", ["AAPL"], snapshot_date="2026-01-15")
        rows = load_index_membership_symbols_as_of("2026-01-15")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], ("us", "AAPL"))

    def test_uses_today_when_date_is_none(self):
        save_index_membership_snapshot("us", ["GOOG"], snapshot_date=None)
        rows = load_index_membership_symbols_as_of(date.today())
        self.assertGreaterEqual(len(rows), 1)

    def test_jp_market_saved(self):
        save_index_membership_snapshot("jp", ["7203", "9984"], snapshot_date="2026-01-15")
        rows = load_index_membership_symbols_as_of("2026-01-15")
        markets = [r[0] for r in rows]
        self.assertIn("jp", markets)


class TestLoadIndexMembershipSymbolsAsOf(_TmpDbTestCase):
    def test_returns_empty_when_no_data(self):
        rows = load_index_membership_symbols_as_of("2026-01-01")
        self.assertEqual(rows, [])

    def test_returns_latest_snapshot_before_date(self):
        save_index_membership_snapshot("us", ["AAPL"], snapshot_date="2026-01-01")
        save_index_membership_snapshot("us", ["AAPL", "GOOG"], snapshot_date="2026-02-01")
        rows = load_index_membership_symbols_as_of("2026-01-15")
        self.assertEqual(len(rows), 1)

    def test_returns_most_recent_snapshot_on_exact_date(self):
        save_index_membership_snapshot("us", ["AAPL"], snapshot_date="2026-03-01")
        rows = load_index_membership_symbols_as_of("2026-03-01")
        self.assertEqual(len(rows), 1)

    def test_multiple_markets_returned(self):
        save_index_membership_snapshot("us", ["AAPL"], snapshot_date="2026-01-01")
        save_index_membership_snapshot("jp", ["7203"], snapshot_date="2026-01-01")
        rows = load_index_membership_symbols_as_of("2026-01-01")
        self.assertEqual(len(rows), 2)
        markets = {r[0] for r in rows}
        self.assertEqual(markets, {"us", "jp"})

    def test_returns_tuples(self):
        save_index_membership_snapshot("us", ["AAPL"], snapshot_date="2026-01-01")
        rows = load_index_membership_symbols_as_of("2026-01-01")
        self.assertIsInstance(rows[0], tuple)
        self.assertEqual(len(rows[0]), 2)
