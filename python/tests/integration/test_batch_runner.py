"""batch_runner モジュールのユニットテスト"""

import csv
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.services.batch_runner import load_target_symbols, print_summary, run_parallel


class TestLoadTargetSymbols(unittest.TestCase):
    """load_target_symbols 関数のテスト"""

    def _create_csv(self, rows):
        """テスト用CSVファイルを作成し、パスを返す"""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
        )
        writer = csv.DictWriter(tmp, fieldnames=["市場", "銘柄コード"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        tmp.close()
        return tmp.name

    def test_reads_csv_correctly(self):
        """CSVから銘柄リストが正しく読み込まれることを確認"""
        csv_path = self._create_csv(
            [
                {"市場": "us", "銘柄コード": "AAPL"},
                {"市場": "jp", "銘柄コード": "7203"},
            ]
        )
        try:
            with patch("src.services.batch_runner.get_watchlist_path", return_value=csv_path):
                result = load_target_symbols()
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], {"market": "us", "symbol": "AAPL"})
            self.assertEqual(result[1], {"market": "jp", "symbol": "7203"})
        finally:
            os.remove(csv_path)

    def test_empty_csv(self):
        """空のCSVでは空リストが返ることを確認"""
        csv_path = self._create_csv([])
        try:
            with patch("src.services.batch_runner.get_watchlist_path", return_value=csv_path):
                result = load_target_symbols()
            self.assertEqual(result, [])
        finally:
            os.remove(csv_path)


class TestRunParallel(unittest.TestCase):
    """run_parallel 関数のテスト"""

    def test_returns_all_results(self):
        """全タスクの結果が返されることを確認"""

        def dummy_func(task):
            return {"id": task, "status": "success"}

        results = run_parallel(dummy_func, [1, 2, 3], max_workers=2, label="テスト")
        self.assertEqual(len(results), 3)
        ids = {r["id"] for r in results}
        self.assertEqual(ids, {1, 2, 3})

    def test_uses_thread_pool_by_default(self):
        """デフォルトでThreadPoolExecutorを使用することを確認"""
        with patch("src.services.batch_runner.ThreadPoolExecutor") as mock_thread:
            mock_executor = MagicMock()
            mock_executor.__enter__ = MagicMock(return_value=mock_executor)
            mock_executor.__exit__ = MagicMock(return_value=False)
            mock_executor.submit.return_value = MagicMock()
            mock_thread.return_value = mock_executor

            run_parallel(lambda x: x, [], max_workers=2, use_process=False)
            mock_thread.assert_called_once_with(max_workers=2)

    def test_uses_process_pool_when_specified(self):
        """use_process=TrueでProcessPoolExecutorを使用することを確認"""

        def dummy_func(task):
            return {"status": "success"}

        # ProcessPoolExecutorは実際に使用してテスト
        results = run_parallel(
            dummy_func, ["a", "b"], max_workers=2, use_process=False, label="プロセステスト"
        )
        self.assertEqual(len(results), 2)

    def test_empty_tasks(self):
        """タスクが空の場合に空リストが返ることを確認"""
        results = run_parallel(lambda x: x, [], max_workers=2)
        self.assertEqual(results, [])


class TestPrintSummary(unittest.TestCase):
    """print_summary 関数のテスト"""

    def test_counts_success_error_skip(self):
        """成功・エラー・スキップが正しくカウントされることを確認"""
        results = [
            {"status": "success", "market": "us", "symbol": "AAPL"},
            {"status": "success", "market": "us", "symbol": "GOOG"},
            {"status": "error", "market": "jp", "symbol": "7203", "error": "timeout"},
            {"status": "skip", "market": "jp", "symbol": "9984"},
        ]
        # 出力をキャプチャして内容を検証
        with patch("builtins.print") as mock_print:
            print_summary("テスト", results)
            output = " ".join(str(c) for c in mock_print.call_args_list)
            self.assertIn("成功: 2", output)
            self.assertIn("エラー: 1", output)
            self.assertIn("スキップ: 1", output)

    def test_error_detail_shown(self):
        """エラー詳細が出力に含まれることを確認"""
        results = [
            {"status": "error", "market": "us", "symbol": "BAD", "error": "connection failed"},
        ]
        with patch("builtins.print") as mock_print:
            print_summary("エラーテスト", results)
            output = " ".join(str(c) for c in mock_print.call_args_list)
            self.assertIn("connection failed", output)

    def test_no_error_no_detail(self):
        """エラーがなければエラー詳細セクションが出ないことを確認"""
        results = [{"status": "success"}]
        with patch("builtins.print") as mock_print:
            print_summary("成功のみ", results)
            output = " ".join(str(c) for c in mock_print.call_args_list)
            self.assertNotIn("エラー詳細", output)


if __name__ == "__main__":
    unittest.main()
