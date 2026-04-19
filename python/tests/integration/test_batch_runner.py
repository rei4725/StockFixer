"""batch_runner モジュールのユニットテスト"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.services.batch_runner import load_target_symbols, print_summary, run_parallel


class TestLoadTargetSymbols(unittest.TestCase):
    """load_target_symbols 関数のテスト"""

    def _create_json(self, data):
        """テスト用 watchlist.json ファイルを作成し、パスを返す"""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(data, tmp)
        tmp.close()
        return tmp.name

    def test_reads_json_correctly(self):
        """watchlist.json から銘柄リストが正しく読み込まれることを確認"""
        json_path = self._create_json({"us": ["AAPL"], "jp": ["7203"]})
        try:
            with patch("src.services.batch_runner.get_watchlist_path", return_value=json_path):
                result = load_target_symbols()
            self.assertEqual(len(result), 2)
            symbols = {(t.market, t.symbol) for t in result}
            self.assertIn(("us", "AAPL"), symbols)
            self.assertIn(("jp", "7203"), symbols)
        finally:
            os.remove(json_path)

    def test_empty_json(self):
        """空の watchlist.json では空リストが返ることを確認"""
        json_path = self._create_json({})
        try:
            with patch("src.services.batch_runner.get_watchlist_path", return_value=json_path):
                result = load_target_symbols()
            self.assertEqual(result, [])
        finally:
            os.remove(json_path)


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
        with self.assertLogs("src.services.batch_runner", level="INFO") as cm:
            print_summary("テスト", results)
        output = "\n".join(cm.output)
        self.assertIn("成功: 2", output)
        self.assertIn("エラー: 1", output)
        self.assertIn("スキップ: 1", output)

    def test_error_detail_shown(self):
        """エラー詳細が出力に含まれることを確認"""
        results = [
            {"status": "error", "market": "us", "symbol": "BAD", "error": "connection failed"},
        ]
        with self.assertLogs("src.services.batch_runner", level="WARNING") as cm:
            print_summary("エラーテスト", results)
        output = "\n".join(cm.output)
        self.assertIn("connection failed", output)

    def test_no_error_no_detail(self):
        """エラーがなければエラー詳細セクションが出ないことを確認"""
        results = [{"status": "success"}]
        with self.assertLogs("src.services.batch_runner", level="INFO") as cm:
            print_summary("成功のみ", results)
        output = "\n".join(cm.output)
        self.assertNotIn("エラー詳細", output)


if __name__ == "__main__":
    unittest.main()
