"""
market_data_raw テーブル操作のユニットテスト

upsert_raw_ohlcv / load_raw_ohlcv / save_raw_ohlcv の動作を確認する。
test_db.py と同じセットアップパターンで一時 DB を使用する。
"""

import os
import tempfile
import unittest

import pandas as pd

import src.utils.data_path_utils as path_utils
import src.utils.db as db_module
from src.utils.db import load_raw_ohlcv, upsert_raw_ohlcv


def _row(
    market="jp",
    symbol="7203",
    ticker="7203.T",
    ts="2024-01-04",
    close=100.0,
    volume=1000,
) -> dict:
    return {
        "market": market,
        "symbol": symbol,
        "ticker": ticker,
        "timeframe": "1d",
        "ts": pd.Timestamp(ts),
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": volume,
        "adj_close": close,
        "source": "yfinance",
    }


class TestMarketDataRawSetup(unittest.TestCase):
    """DB セットアップ・テーブル存在確認"""

    def setUp(self):
        db_module.close_connection()
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_db = os.path.join(self.tmp_dir, "test_raw.duckdb")
        self._orig = path_utils.get_db_path
        path_utils.get_db_path = lambda: self.tmp_db
        db_module.get_db_path = lambda: self.tmp_db
        db_module._tables_initialized = False

    def tearDown(self):
        db_module.close_connection()
        path_utils.get_db_path = self._orig
        db_module.get_db_path = self._orig
        if os.path.exists(self.tmp_db):
            os.remove(self.tmp_db)
        wal = self.tmp_db + ".wal"
        if os.path.exists(wal):
            os.remove(wal)
        if os.path.exists(self.tmp_dir):
            os.rmdir(self.tmp_dir)

    def test_market_data_raw_table_exists_after_init(self):
        with db_module._db_connection() as con:
            tables = con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        names = {r[0] for r in tables}
        self.assertIn("market_data_raw", names)

    def test_all_three_tables_created(self):
        with db_module._db_connection() as con:
            tables = con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        names = {r[0] for r in tables}
        self.assertIn("stock_features", names)
        self.assertIn("prediction_results", names)
        self.assertIn("market_data_raw", names)


class TestUpsertRawOhlcv(unittest.TestCase):
    """upsert_raw_ohlcv のテスト"""

    def setUp(self):
        db_module.close_connection()
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_db = os.path.join(self.tmp_dir, "test_raw.duckdb")
        self._orig = path_utils.get_db_path
        path_utils.get_db_path = lambda: self.tmp_db
        db_module.get_db_path = lambda: self.tmp_db
        db_module._tables_initialized = False

    def tearDown(self):
        db_module.close_connection()
        path_utils.get_db_path = self._orig
        db_module.get_db_path = self._orig
        if os.path.exists(self.tmp_db):
            os.remove(self.tmp_db)
        wal = self.tmp_db + ".wal"
        if os.path.exists(wal):
            os.remove(wal)
        if os.path.exists(self.tmp_dir):
            os.rmdir(self.tmp_dir)

    def test_upsert_empty_list_returns_zero(self):
        n = upsert_raw_ohlcv([])
        self.assertEqual(n, 0)

    def test_upsert_single_row_returns_one(self):
        n = upsert_raw_ohlcv([_row()])
        self.assertEqual(n, 1)

    def test_upsert_multiple_rows(self):
        rows = [_row(ts=f"2024-01-0{i+1}", close=100 + i) for i in range(3)]
        n = upsert_raw_ohlcv(rows)
        self.assertEqual(n, 3)

    def test_upsert_idempotent(self):
        """同一PK のUPSERT は上書きで重複しない"""
        upsert_raw_ohlcv([_row(ts="2024-01-04", close=100.0)])
        upsert_raw_ohlcv([_row(ts="2024-01-04", close=110.0)])
        df = load_raw_ohlcv("jp", "7203")
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(float(df["Close"].iloc[0]), 110.0)

    def test_upsert_without_adj_close(self):
        """adj_close が省略されてもエラーにならない"""
        row = _row()
        del row["adj_close"]
        n = upsert_raw_ohlcv([row])
        self.assertEqual(n, 1)

    def test_upsert_without_source_defaults_to_yfinance(self):
        row = _row()
        del row["source"]
        upsert_raw_ohlcv([row])
        with db_module._db_connection() as con:
            src = con.execute(
                "SELECT source FROM market_data_raw WHERE market='jp' AND symbol='7203'"
            ).fetchone()
        self.assertIsNotNone(src)


class TestLoadRawOhlcv(unittest.TestCase):
    """load_raw_ohlcv のテスト"""

    def setUp(self):
        db_module.close_connection()
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_db = os.path.join(self.tmp_dir, "test_raw.duckdb")
        self._orig = path_utils.get_db_path
        path_utils.get_db_path = lambda: self.tmp_db
        db_module.get_db_path = lambda: self.tmp_db
        db_module._tables_initialized = False
        # 初期データ挿入
        rows = [
            _row(ts="2024-01-02", close=100.0),
            _row(ts="2024-01-03", close=102.0),
            _row(ts="2024-01-04", close=101.0),
            _row(ts="2024-01-05", close=105.0),
        ]
        upsert_raw_ohlcv(rows)

    def tearDown(self):
        db_module.close_connection()
        path_utils.get_db_path = self._orig
        db_module.get_db_path = self._orig
        if os.path.exists(self.tmp_db):
            os.remove(self.tmp_db)
        wal = self.tmp_db + ".wal"
        if os.path.exists(wal):
            os.remove(wal)
        if os.path.exists(self.tmp_dir):
            os.rmdir(self.tmp_dir)

    def test_load_returns_dataframe(self):
        df = load_raw_ohlcv("jp", "7203")
        self.assertIsInstance(df, pd.DataFrame)

    def test_load_all_rows(self):
        df = load_raw_ohlcv("jp", "7203")
        self.assertEqual(len(df), 4)

    def test_load_returns_none_for_missing_symbol(self):
        result = load_raw_ohlcv("jp", "9999")
        self.assertIsNone(result)

    def test_load_date_range_filter(self):
        df = load_raw_ohlcv("jp", "7203", start_date="2024-01-03", end_date="2024-01-04")
        self.assertEqual(len(df), 2)

    def test_load_start_date_only(self):
        df = load_raw_ohlcv("jp", "7203", start_date="2024-01-04")
        self.assertEqual(len(df), 2)  # 01-04, 01-05

    def test_load_end_date_only(self):
        df = load_raw_ohlcv("jp", "7203", end_date="2024-01-03")
        self.assertEqual(len(df), 2)  # 01-02, 01-03

    def test_load_columns_yfinance_style(self):
        """返却 DataFrame のカラム名が yfinance 形式（先頭大文字）になっている"""
        df = load_raw_ohlcv("jp", "7203")
        self.assertIn("Close", df.columns)
        self.assertIn("Open", df.columns)
        self.assertIn("High", df.columns)
        self.assertIn("Low", df.columns)
        self.assertIn("Volume", df.columns)

    def test_load_index_is_date(self):
        """インデックスが日付（Date）である"""
        df = load_raw_ohlcv("jp", "7203")
        self.assertEqual(df.index.name, "Date")

    def test_load_sorted_by_date(self):
        """日付昇順で返る"""
        df = load_raw_ohlcv("jp", "7203")
        self.assertTrue(df.index.is_monotonic_increasing)

    def test_load_different_market_returns_none(self):
        result = load_raw_ohlcv("us", "7203")
        self.assertIsNone(result)


class TestSaveRawOhlcv(unittest.TestCase):
    """data_saver.save_raw_ohlcv のテスト"""

    def setUp(self):
        db_module.close_connection()
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_db = os.path.join(self.tmp_dir, "test_raw.duckdb")
        self._orig = path_utils.get_db_path
        path_utils.get_db_path = lambda: self.tmp_db
        db_module.get_db_path = lambda: self.tmp_db
        db_module._tables_initialized = False

    def tearDown(self):
        db_module.close_connection()
        path_utils.get_db_path = self._orig
        db_module.get_db_path = self._orig
        if os.path.exists(self.tmp_db):
            os.remove(self.tmp_db)
        wal = self.tmp_db + ".wal"
        if os.path.exists(wal):
            os.remove(wal)
        if os.path.exists(self.tmp_dir):
            os.rmdir(self.tmp_dir)

    def test_save_raw_ohlcv_from_yfinance_df(self):
        """yfinance 形式のDataFrameをそのまま保存できる"""
        from src.data.data_saver import save_raw_ohlcv

        idx = pd.date_range("2024-01-02", periods=3, freq="B")
        df = pd.DataFrame(
            {
                "Open": [99.0, 100.0, 101.0],
                "High": [101.0, 102.0, 103.0],
                "Low": [98.0, 99.0, 100.0],
                "Close": [100.0, 101.0, 102.0],
                "Volume": [1000, 2000, 1500],
            },
            index=idx,
        )

        n = save_raw_ohlcv("jp", "7203", df)
        self.assertEqual(n, 3)
        loaded = load_raw_ohlcv("jp", "7203")
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded), 3)

    def test_save_raw_ohlcv_returns_row_count(self):
        from src.data.data_saver import save_raw_ohlcv

        idx = pd.date_range("2024-02-01", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "Open": [100.0] * 5,
                "High": [105.0] * 5,
                "Low": [95.0] * 5,
                "Close": [102.0] * 5,
                "Volume": [1000] * 5,
            },
            index=idx,
        )
        n = save_raw_ohlcv("us", "AAPL", df)
        self.assertEqual(n, 5)


if __name__ == "__main__":
    unittest.main()
