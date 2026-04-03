"""db/market_data モジュールのユニットテスト（一時DB使用）"""

import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import src.utils.data_path_utils as path_utils
import src.utils.db as db_module
from src.utils.db.market_data import load_all_raw_ohlcv_symbols, load_raw_ohlcv, upsert_raw_ohlcv


class _TmpDbTestCase(unittest.TestCase):
    """一時DBを使うテストの共通基底クラス"""

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
        for path in (self.tmp_db, self.tmp_db + ".wal"):
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(self.tmp_dir):
            os.rmdir(self.tmp_dir)

    def _make_rows(self, market="us", symbol="AAPL", n=3):
        return [
            {
                "market": market,
                "symbol": symbol,
                "ticker": symbol if market == "us" else symbol + ".T",
                "timeframe": "1d",
                "ts": pd.Timestamp(f"2026-01-{i+1:02d}"),
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "volume": 1000 * (i + 1),
            }
            for i in range(n)
        ]


class TestUpsertRawOhlcv(_TmpDbTestCase):
    """upsert_raw_ohlcv のテスト"""

    def test_returns_row_count(self):
        """保存した行数を返すこと"""
        rows = self._make_rows(n=3)
        count = upsert_raw_ohlcv(rows)
        self.assertEqual(count, 3)

    def test_empty_rows_returns_zero(self):
        """空リストは0を返すこと"""
        count = upsert_raw_ohlcv([])
        self.assertEqual(count, 0)

    def test_upsert_is_idempotent(self):
        """同じデータを2回保存しても重複しないこと"""
        rows = self._make_rows(n=2)
        upsert_raw_ohlcv(rows)
        upsert_raw_ohlcv(rows)

        df = load_raw_ohlcv("us", "AAPL")
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)

    def test_adj_close_defaults_to_none(self):
        """adj_close を省略しても保存が成功すること"""
        rows = self._make_rows(n=1)
        # adj_close なし
        rows[0].pop("adj_close", None)
        count = upsert_raw_ohlcv(rows)
        self.assertEqual(count, 1)

    def test_source_defaults_to_yfinance(self):
        """source を省略したとき yfinance がデフォルトになること"""
        rows = self._make_rows(n=1)
        rows[0].pop("source", None)
        upsert_raw_ohlcv(rows)
        df = load_raw_ohlcv("us", "AAPL")
        self.assertIsNotNone(df)

    def test_multiple_symbols_saved(self):
        """複数銘柄を保存できること"""
        upsert_raw_ohlcv(self._make_rows("us", "AAPL", 2))
        upsert_raw_ohlcv(self._make_rows("us", "MSFT", 3))

        df_aapl = load_raw_ohlcv("us", "AAPL")
        df_msft = load_raw_ohlcv("us", "MSFT")
        self.assertEqual(len(df_aapl), 2)
        self.assertEqual(len(df_msft), 3)


class TestLoadRawOhlcv(_TmpDbTestCase):
    """load_raw_ohlcv のテスト"""

    def setUp(self):
        super().setUp()
        rows = self._make_rows("us", "AAPL", 5)
        upsert_raw_ohlcv(rows)

    def test_returns_dataframe(self):
        """DataFrameが返ること"""
        df = load_raw_ohlcv("us", "AAPL")
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)

    def test_returns_none_when_no_data(self):
        """データがない場合はNoneを返すこと"""
        df = load_raw_ohlcv("us", "UNKNOWN")
        self.assertIsNone(df)

    def test_columns_capitalized(self):
        """カラム名が先頭大文字（yfinance形式）に変換されること"""
        df = load_raw_ohlcv("us", "AAPL")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            self.assertIn(col, df.columns)

    def test_index_is_date(self):
        """インデックスがDaemon 'Date' であること"""
        df = load_raw_ohlcv("us", "AAPL")
        self.assertEqual(df.index.name, "Date")

    def test_start_date_filter(self):
        """start_date フィルタが機能すること"""
        df = load_raw_ohlcv("us", "AAPL", start_date="2026-01-03")
        self.assertIsNotNone(df)
        self.assertLessEqual(len(df), 5)
        self.assertTrue(all(df.index >= pd.Timestamp("2026-01-03")))

    def test_end_date_filter(self):
        """end_date フィルタが機能すること"""
        df = load_raw_ohlcv("us", "AAPL", end_date="2026-01-02")
        self.assertIsNotNone(df)
        self.assertTrue(all(df.index <= pd.Timestamp("2026-01-02")))

    def test_start_and_end_date_filter(self):
        """start_date + end_date 両方指定が機能すること"""
        df = load_raw_ohlcv("us", "AAPL", start_date="2026-01-02", end_date="2026-01-03")
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)

    def test_timeframe_filter(self):
        """timeframe フィルタが機能すること（存在しないtimeframeはNone）"""
        df = load_raw_ohlcv("us", "AAPL", timeframe="1h")
        self.assertIsNone(df)

    def test_rows_sorted_ascending(self):
        """行が日付昇順でソートされること"""
        df = load_raw_ohlcv("us", "AAPL")
        self.assertTrue(df.index.is_monotonic_increasing)


class TestLoadAllRawOhlcvSymbols(_TmpDbTestCase):
    """load_all_raw_ohlcv_symbols のテスト"""

    def test_returns_empty_when_no_data(self):
        """データがない場合は空リストを返すこと"""
        result = load_all_raw_ohlcv_symbols()
        self.assertEqual(result, [])

    def test_returns_all_symbol_pairs(self):
        """全 (market, symbol) ペアが返ること"""
        upsert_raw_ohlcv(self._make_rows("us", "AAPL", 2))
        upsert_raw_ohlcv(self._make_rows("jp", "7203", 2))

        result = load_all_raw_ohlcv_symbols()
        self.assertIn(("us", "AAPL"), result)
        self.assertIn(("jp", "7203"), result)

    def test_no_duplicates(self):
        """同じ銘柄を重複なく返すこと"""
        upsert_raw_ohlcv(self._make_rows("us", "AAPL", 5))
        result = load_all_raw_ohlcv_symbols()
        self.assertEqual(result.count(("us", "AAPL")), 1)

    def test_filters_by_timeframe(self):
        """timeframe フィルタが機能すること"""
        rows_1h = [
            {
                "market": "us",
                "symbol": "AAPL",
                "ticker": "AAPL",
                "timeframe": "1h",
                "ts": pd.Timestamp("2026-01-01"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000,
            }
        ]
        upsert_raw_ohlcv(rows_1h)
        upsert_raw_ohlcv(self._make_rows("us", "MSFT", 2))

        result_1d = load_all_raw_ohlcv_symbols(timeframe="1d")
        result_1h = load_all_raw_ohlcv_symbols(timeframe="1h")

        self.assertIn(("us", "MSFT"), result_1d)
        self.assertNotIn(("us", "MSFT"), result_1h)
        self.assertIn(("us", "AAPL"), result_1h)

    def test_sorted_by_market_symbol(self):
        """market, symbol でソートされること"""
        upsert_raw_ohlcv(self._make_rows("us", "MSFT", 2))
        upsert_raw_ohlcv(self._make_rows("jp", "7203", 2))
        upsert_raw_ohlcv(self._make_rows("us", "AAPL", 2))

        result = load_all_raw_ohlcv_symbols()
        self.assertEqual(result, sorted(result))

    def test_returns_empty_on_db_error(self):
        """DBエラー時は空リストを返すこと"""
        with patch("src.utils.db.market_data._db_connection") as mock_conn:
            mock_conn.return_value.__enter__.return_value.execute.side_effect = Exception(
                "DB error"
            )
            result = load_all_raw_ohlcv_symbols()
        self.assertEqual(result, [])

    def test_load_raw_ohlcv_returns_none_on_db_error(self):
        """load_raw_ohlcv でDBエラー時はNoneを返すこと"""
        with patch("src.utils.db.market_data._db_connection") as mock_conn:
            mock_conn.return_value.__enter__.return_value.execute.side_effect = Exception(
                "DB error"
            )
            result = load_raw_ohlcv("us", "AAPL")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
