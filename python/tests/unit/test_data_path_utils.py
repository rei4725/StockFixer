"""data_path_utils モジュールのユニットテスト"""

import os
import tempfile
import unittest

from src.utils.data_path_utils import (
    ensure_dir,
    get_data_dir,
    get_data_subdir,
    get_db_path,
    get_model_path,
    get_models_dir,
    get_models_subdir,
    get_monitor_list_path,
    get_python_root,
    get_results_dir,
    get_results_subdir,
    get_ticker,
    get_watchlist_path,
)


class TestDataPathUtils(unittest.TestCase):
    """パスユーティリティ関数のテスト"""

    def test_get_python_root_is_absolute(self):
        """get_python_root が絶対パスを返すことを確認"""
        root = get_python_root()
        self.assertTrue(os.path.isabs(root))

    def test_get_data_dir_under_python_root(self):
        """get_data_dir が python_root/data を返すことを確認"""
        self.assertEqual(get_data_dir(), os.path.join(get_python_root(), "data"))

    def test_get_data_subdir_format(self):
        """get_data_subdir が {market}_{symbol} 形式のサブディレクトリを返すことを確認"""
        path = get_data_subdir("us", "AAPL")
        self.assertTrue(path.endswith(os.path.join("data", "us_AAPL")))

    def test_get_models_dir_under_python_root(self):
        """get_models_dir が python_root/models を返すことを確認"""
        self.assertEqual(get_models_dir(), os.path.join(get_python_root(), "models"))

    def test_get_models_subdir_format(self):
        """get_models_subdir が {market}_{symbol} 形式のサブディレクトリを返すことを確認"""
        path = get_models_subdir("jp", "7203")
        self.assertTrue(path.endswith(os.path.join("models", "jp_7203")))

    def test_get_model_path_format(self):
        """get_model_path が .joblib 拡張子のパスを返すことを確認"""
        path = get_model_path("us", "AAPL", "StockXGBoostModel")
        self.assertTrue(path.endswith("StockXGBoostModel.joblib"))
        self.assertIn("us_AAPL", path)

    def test_get_results_dir_under_python_root(self):
        """get_results_dir が python_root/results を返すことを確認"""
        self.assertEqual(get_results_dir(), os.path.join(get_python_root(), "results"))

    def test_get_results_subdir_with_timestamp(self):
        """get_results_subdir にタイムスタンプを渡すとそのサブディレクトリを返すことを確認"""
        ts = "20260101_120000"
        path = get_results_subdir(ts)
        self.assertTrue(path.endswith(ts))

    def test_get_results_subdir_auto_timestamp(self):
        """get_results_subdir にNoneを渡すと自動生成タイムスタンプのパスを返すことを確認"""
        path = get_results_subdir(None)
        # YYYYMMDD_HHMMSS 形式（15文字）がディレクトリ名になっている
        dirname = os.path.basename(path)
        self.assertEqual(len(dirname), 15)

    def test_get_watchlist_path_ends_with_csv(self):
        """get_watchlist_path がJSONファイルパスを返すことを確認"""
        self.assertTrue(get_watchlist_path().endswith(".json"))

    def test_get_monitor_list_path_ends_with_csv(self):
        """get_monitor_list_path がCSVファイルパスを返すことを確認"""
        self.assertTrue(get_monitor_list_path().endswith(".csv"))

    def test_get_db_path_ends_with_duckdb(self):
        """get_db_path がDuckDBファイルパスを返すことを確認"""
        self.assertTrue(get_db_path().endswith("stockfixer.duckdb"))


class TestGetTicker(unittest.TestCase):
    """get_ticker 関数のテスト"""

    def test_jp_market_adds_suffix(self):
        """日本株シンボルに .T が付与されることを確認"""
        self.assertEqual(get_ticker("jp", "7203"), "7203.T")

    def test_jp_market_no_double_suffix(self):
        """既に .T が付いているシンボルに二重付与されないことを確認"""
        self.assertEqual(get_ticker("jp", "7203.T"), "7203.T")

    def test_japan_alias(self):
        """'japan' もJP市場として扱われることを確認"""
        self.assertEqual(get_ticker("japan", "7203"), "7203.T")

    def test_us_market_unchanged(self):
        """US市場のシンボルがそのまま返されることを確認"""
        self.assertEqual(get_ticker("us", "AAPL"), "AAPL")

    def test_us_aliases(self):
        """US市場の別名（usa, nyse, nasdaq）がそのまま返されることを確認"""
        for m in ["usa", "nyse", "nasdaq"]:
            self.assertEqual(get_ticker(m, "MSFT"), "MSFT")

    def test_unknown_market_unchanged(self):
        """未知の市場のシンボルがそのまま返されることを確認"""
        self.assertEqual(get_ticker("kr", "005930"), "005930")

    def test_case_insensitive_market(self):
        """市場名が大文字でも正しく処理されることを確認"""
        self.assertEqual(get_ticker("JP", "7203"), "7203.T")
        self.assertEqual(get_ticker("US", "AAPL"), "AAPL")


class TestEnsureDir(unittest.TestCase):
    """ensure_dir 関数のテスト"""

    def test_creates_directory(self):
        """存在しないディレクトリが作成されることを確認"""
        with tempfile.TemporaryDirectory() as tmp:
            new_dir = os.path.join(tmp, "new_subdir")
            result = ensure_dir(new_dir)
            self.assertTrue(os.path.isdir(new_dir))
            self.assertEqual(result, new_dir)

    def test_existing_directory_no_error(self):
        """既存ディレクトリに対して例外が発生しないことを確認"""
        with tempfile.TemporaryDirectory() as tmp:
            result = ensure_dir(tmp)
            self.assertEqual(result, tmp)

    def test_nested_directory(self):
        """ネストされたディレクトリが再帰的に作成されることを確認"""
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "a", "b", "c")
            ensure_dir(nested)
            self.assertTrue(os.path.isdir(nested))


if __name__ == "__main__":
    unittest.main()
