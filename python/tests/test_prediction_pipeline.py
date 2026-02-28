"""prediction_pipeline モジュールのユニットテスト"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

import src.utils.db as db_module
import src.utils.data_path_utils as path_utils
from src.services.prediction_pipeline import (
    find_model_files,
    output_top_worst_results,
)


class TestFindModelFiles(unittest.TestCase):
    """find_model_files 関数のテスト"""

    def setUp(self):
        """テスト用ディレクトリにダミーモデルファイルを作成"""
        self.tmp_dir = tempfile.mkdtemp()
        # us_AAPL/StockXGBoostModel.joblib
        aapl_dir = os.path.join(self.tmp_dir, "us_AAPL")
        os.makedirs(aapl_dir)
        open(os.path.join(aapl_dir, "StockXGBoostModel.joblib"), "w").close()
        # jp_7203/StockXGBoostModel.joblib
        jp_dir = os.path.join(self.tmp_dir, "jp_7203")
        os.makedirs(jp_dir)
        open(os.path.join(jp_dir, "StockXGBoostModel.joblib"), "w").close()
        # invalid_dir（_なし）は無視される
        bad_dir = os.path.join(self.tmp_dir, "invalid")
        os.makedirs(bad_dir)
        open(os.path.join(bad_dir, "StockXGBoostModel.joblib"), "w").close()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir)

    def test_finds_correct_model_files(self):
        """market_symbol形式のディレクトリからモデルが正しく検出されることを確認"""
        results = find_model_files(model_root=self.tmp_dir, model_name="StockXGBoostModel.joblib")
        markets = {(m, s) for m, s, _ in results}
        self.assertIn(("us", "AAPL"), markets)
        self.assertIn(("jp", "7203"), markets)
        self.assertEqual(len(results), 2)

    def test_no_match(self):
        """存在しないモデル名では空リストが返ることを確認"""
        results = find_model_files(model_root=self.tmp_dir, model_name="NonExistent.joblib")
        self.assertEqual(results, [])

    def test_empty_dir(self):
        """空ディレクトリでは空リストが返ることを確認"""
        empty = tempfile.mkdtemp()
        try:
            results = find_model_files(model_root=empty)
            self.assertEqual(results, [])
        finally:
            os.rmdir(empty)


class TestOutputTopWorstResults(unittest.TestCase):
    """output_top_worst_results 関数のテスト"""

    def setUp(self):
        """テスト用一時DBを設定"""
        db_module.close_connection()
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_db = os.path.join(self.tmp_dir, "test_pred_pipeline.duckdb")
        self._orig_get_db_path = path_utils.get_db_path
        path_utils.get_db_path = lambda: self.tmp_db
        db_module.get_db_path = lambda: self.tmp_db
        db_module._connection = None

    def tearDown(self):
        db_module.close_connection()
        path_utils.get_db_path = self._orig_get_db_path
        db_module.get_db_path = self._orig_get_db_path
        if os.path.exists(self.tmp_db):
            os.remove(self.tmp_db)
        wal_path = self.tmp_db + ".wal"
        if os.path.exists(wal_path):
            os.remove(wal_path)
        if os.path.exists(self.tmp_dir):
            os.rmdir(self.tmp_dir)

    def _make_rows(self, n=15, market="us"):
        """テスト用予測結果DataFrameリストを生成する"""
        rows = []
        for i in range(n):
            rows.append(pd.DataFrame([{
                "market": market,
                "symbol": f"SYM{i:03d}",
                "current_price": 100.0 + i,
                "avg_pred_price": 100.0 + i + (i - n // 2),
                "diff_ratio": (i - n // 2) / 100.0,
                "model_count": 2,
            }]))
        return rows

    def test_empty_rows(self):
        """空リストが渡された場合にエラーにならないことを確認"""
        # 例外が発生しなければ成功
        output_top_worst_results([], mode="test")

    def test_saves_to_db(self):
        """結果がDBに保存されることを確認"""
        rows = self._make_rows(25, market="us")
        output_top_worst_results(rows, mode="individual")

        # prediction_resultsテーブルにデータが保存されている
        from src.utils.db import load_prediction_results
        df = load_prediction_results()
        self.assertFalse(df.empty)
        # 全銘柄が保存される
        self.assertEqual(len(df), 25)

    def test_top10_worst10_correct_order(self):
        """Top10が降順、Worst10が昇順で取得できることを確認"""
        rows = self._make_rows(20, market="us")
        output_top_worst_results(rows, mode="unified")

        from src.utils.db import load_prediction_results
        df_top = load_prediction_results(market="us", top_n=10)
        df_worst = load_prediction_results(market="us", worst_n=10)

        if not df_top.empty:
            # Top10のdiff_ratioは降順
            self.assertGreaterEqual(
                df_top["diff_ratio"].iloc[0],
                df_top["diff_ratio"].iloc[-1]
            )
        if not df_worst.empty:
            # Worst10のdiff_ratioは昇順
            self.assertLessEqual(
                df_worst["diff_ratio"].iloc[0],
                df_worst["diff_ratio"].iloc[-1]
            )

    def test_multiple_markets(self):
        """複数市場のデータが正しく分離されて保存されることを確認"""
        rows = self._make_rows(25, market="us") + self._make_rows(25, market="jp")
        output_top_worst_results(rows, mode="individual")

        from src.utils.db import load_prediction_results, load_prediction_markets
        markets = load_prediction_markets()
        self.assertIn("us", markets)
        self.assertIn("jp", markets)


if __name__ == '__main__':
    unittest.main()
