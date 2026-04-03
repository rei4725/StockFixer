"""db/stock_features モジュールのユニットテスト（一時DB使用）"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import src.utils.data_path_utils as path_utils
import src.utils.db as db_module
from src.utils.db.stock_features import (
    delete_stock_features,
    get_all_symbols,
    load_all_stock_features,
    load_stock_features,
    upsert_stock_features,
)


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

    def _make_df(self, n=5, extra_cols=None):
        """DatetimeIndex付きの簡単な特徴量DataFrameを作成する"""
        idx = pd.date_range("2026-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "close": np.random.uniform(100, 200, n),
                "volume": np.random.randint(1000, 5000, n),
            },
            index=idx,
        )
        if extra_cols:
            for col, dtype in extra_cols.items():
                if dtype == "int":
                    df[col] = np.random.randint(0, 10, n)
                elif dtype == "bool":
                    df[col] = np.random.choice([True, False], n)
                elif dtype == "str":
                    df[col] = "value"
                elif dtype == "datetime":
                    df[col] = pd.date_range("2026-01-01", periods=n, freq="B")
                else:
                    df[col] = np.random.uniform(0, 1, n)
        return df


class TestUpsertStockFeatures(_TmpDbTestCase):
    """upsert_stock_features のテスト"""

    def test_upsert_basic(self):
        """基本的な保存が成功すること"""
        df = self._make_df()
        upsert_stock_features("us", "AAPL", df)

    def test_upsert_and_load_roundtrip(self):
        """保存したデータを読み込めること"""
        df = self._make_df(5)
        upsert_stock_features("us", "AAPL", df)

        result = load_stock_features("us", "AAPL")
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5)

    def test_upsert_is_idempotent(self):
        """同じデータを2回保存しても重複しないこと（べき等性）"""
        df = self._make_df(3)
        upsert_stock_features("us", "AAPL", df)
        upsert_stock_features("us", "AAPL", df)

        result = load_stock_features("us", "AAPL")
        self.assertEqual(len(result), 3)

    def test_upsert_with_integer_column(self):
        """整数型列が正しく保存されること（_ensure_columns の BIGINT 分岐）"""
        df = self._make_df(3, extra_cols={"int_col": "int"})
        upsert_stock_features("us", "AAPL", df)

        result = load_stock_features("us", "AAPL")
        self.assertIn("int_col", result.columns)

    def test_upsert_with_bool_column(self):
        """ブール型列が正しく保存されること（_ensure_columns の BOOLEAN 分岐）"""
        df = self._make_df(3, extra_cols={"flag_col": "bool"})
        upsert_stock_features("us", "AAPL", df)

        result = load_stock_features("us", "AAPL")
        self.assertIn("flag_col", result.columns)

    def test_upsert_with_string_column(self):
        """文字列型列が正しく保存されること（_ensure_columns の VARCHAR 分岐）"""
        df = self._make_df(3, extra_cols={"label_col": "str"})
        upsert_stock_features("us", "AAPL", df)

        result = load_stock_features("us", "AAPL")
        self.assertIn("label_col", result.columns)

    def test_upsert_with_datetime_column(self):
        """日時型列が正しく保存されること（_ensure_columns の TIMESTAMP 分岐）"""
        df = self._make_df(3, extra_cols={"ts_col": "datetime"})
        upsert_stock_features("us", "AAPL", df)

        result = load_stock_features("us", "AAPL")
        self.assertIn("ts_col", result.columns)

    def test_multiple_symbols_independent(self):
        """異なる銘柄が独立して保存されること"""
        df_aapl = self._make_df(3)
        df_msft = self._make_df(4)
        upsert_stock_features("us", "AAPL", df_aapl)
        upsert_stock_features("us", "MSFT", df_msft)

        result_aapl = load_stock_features("us", "AAPL")
        result_msft = load_stock_features("us", "MSFT")
        self.assertEqual(len(result_aapl), 3)
        self.assertEqual(len(result_msft), 4)

    def test_upsert_drops_market_symbol_rownum_columns(self):
        """読み込み結果にmarket/symbol/row_num列が含まれないこと"""
        df = self._make_df(3)
        upsert_stock_features("us", "AAPL", df)

        result = load_stock_features("us", "AAPL")
        for col in ["market", "symbol", "row_num"]:
            self.assertNotIn(col, result.columns)


class TestLoadStockFeatures(_TmpDbTestCase):
    """load_stock_features のテスト"""

    def test_returns_none_when_no_data(self):
        """データがない場合はNoneを返すこと"""
        result = load_stock_features("us", "UNKNOWN")
        self.assertIsNone(result)

    def test_preserves_row_order(self):
        """行の順序が保持されること"""
        df = self._make_df(5)
        df["close"] = [100.0, 200.0, 150.0, 175.0, 125.0]
        upsert_stock_features("us", "AAPL", df)

        result = load_stock_features("us", "AAPL")
        self.assertEqual(list(result["close"]), [100.0, 200.0, 150.0, 175.0, 125.0])


class TestLoadAllStockFeatures(_TmpDbTestCase):
    """load_all_stock_features のテスト"""

    def test_returns_empty_when_no_data(self):
        """データがない場合は空DataFrameを返すこと"""
        df = load_all_stock_features()
        self.assertTrue(df.empty)

    def test_returns_all_symbols(self):
        """複数銘柄を全件取得できること"""
        upsert_stock_features("us", "AAPL", self._make_df(3))
        upsert_stock_features("us", "MSFT", self._make_df(2))
        upsert_stock_features("jp", "7203", self._make_df(4))

        df = load_all_stock_features()
        self.assertEqual(len(df), 9)

    def test_contains_market_symbol_columns(self):
        """market/symbol 列が含まれること"""
        upsert_stock_features("us", "AAPL", self._make_df(2))
        df = load_all_stock_features()
        self.assertIn("market", df.columns)
        self.assertIn("symbol", df.columns)

    def test_row_num_not_in_result(self):
        """row_num 列が結果に含まれないこと"""
        upsert_stock_features("us", "AAPL", self._make_df(2))
        df = load_all_stock_features()
        self.assertNotIn("row_num", df.columns)


class TestDeleteStockFeatures(_TmpDbTestCase):
    """delete_stock_features のテスト"""

    def test_delete_removes_data(self):
        """削除後はNoneが返ること"""
        upsert_stock_features("us", "AAPL", self._make_df(3))
        delete_stock_features("us", "AAPL")
        result = load_stock_features("us", "AAPL")
        self.assertIsNone(result)

    def test_delete_does_not_affect_other_symbols(self):
        """削除が他の銘柄に影響しないこと"""
        upsert_stock_features("us", "AAPL", self._make_df(3))
        upsert_stock_features("us", "MSFT", self._make_df(2))
        delete_stock_features("us", "AAPL")

        result_msft = load_stock_features("us", "MSFT")
        self.assertIsNotNone(result_msft)
        self.assertEqual(len(result_msft), 2)

    def test_delete_nonexistent_no_error(self):
        """存在しないデータを削除してもエラーにならないこと"""
        delete_stock_features("us", "UNKNOWN")


class TestGetAllSymbols(_TmpDbTestCase):
    """get_all_symbols のテスト"""

    def test_returns_empty_when_no_data(self):
        """データがない場合は空リストを返すこと"""
        result = get_all_symbols()
        self.assertEqual(result, [])

    def test_returns_all_market_symbol_pairs(self):
        """全銘柄の (market, symbol) ペアが返ること"""
        upsert_stock_features("us", "AAPL", self._make_df(2))
        upsert_stock_features("us", "MSFT", self._make_df(2))
        upsert_stock_features("jp", "7203", self._make_df(2))

        result = get_all_symbols()
        self.assertIn(("jp", "7203"), result)
        self.assertIn(("us", "AAPL"), result)
        self.assertIn(("us", "MSFT"), result)
        self.assertEqual(len(result), 3)

    def test_no_duplicates(self):
        """重複して保存しても銘柄は1件のみ返ること"""
        upsert_stock_features("us", "AAPL", self._make_df(2))
        upsert_stock_features("us", "AAPL", self._make_df(3))

        result = get_all_symbols()
        self.assertEqual(result.count(("us", "AAPL")), 1)

    def test_sorted_by_market_symbol(self):
        """market, symbol でソートされて返ること"""
        upsert_stock_features("us", "MSFT", self._make_df(2))
        upsert_stock_features("jp", "7203", self._make_df(2))
        upsert_stock_features("us", "AAPL", self._make_df(2))

        result = get_all_symbols()
        self.assertEqual(result, sorted(result))


if __name__ == "__main__":
    unittest.main()
