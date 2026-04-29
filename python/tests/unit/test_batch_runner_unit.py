"""batch_runner モジュールのユニットテスト"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from src.watchlist.batch_runner import load_target_symbols, print_summary, run_parallel
from src.watchlist.types import SymbolTask


class TestLoadTargetSymbols(unittest.TestCase):
    """load_target_symbols 関数のテスト"""

    def _write_watchlist(self, tmp_dir: str, data: dict) -> str:
        path = os.path.join(tmp_dir, "watchlist.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return path

    @patch("src.watchlist.batch_runner.get_watchlist_path")
    def test_returns_symbol_task_list(self, mock_path):
        """JSON から SymbolTask リストが生成されること"""
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write_watchlist(tmp, {"us": ["AAPL", "MSFT"], "jp": ["7203"]})
            mock_path.return_value = p

            result = load_target_symbols()

        self.assertEqual(len(result), 3)
        self.assertTrue(all(isinstance(t, SymbolTask) for t in result))

    @patch("src.watchlist.batch_runner.get_watchlist_path")
    def test_markets_correctly_assigned(self, mock_path):
        """各 SymbolTask に正しい market が設定されること"""
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write_watchlist(tmp, {"us": ["AAPL"], "jp": ["7203"]})
            mock_path.return_value = p

            result = load_target_symbols()

        markets = {t.market for t in result}
        self.assertIn("us", markets)
        self.assertIn("jp", markets)

    @patch("src.watchlist.batch_runner.get_watchlist_path")
    def test_symbol_correctly_assigned(self, mock_path):
        """各 SymbolTask に正しい symbol が設定されること"""
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write_watchlist(tmp, {"jp": ["9984"]})
            mock_path.return_value = p

            result = load_target_symbols()

        self.assertEqual(result[0].market, "jp")
        self.assertEqual(result[0].symbol, "9984")

    @patch("src.watchlist.batch_runner.get_watchlist_path")
    def test_empty_market_returns_empty_list(self, mock_path):
        """銘柄なしマーケットは空リストを返すこと"""
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write_watchlist(tmp, {"us": []})
            mock_path.return_value = p

            result = load_target_symbols()

        self.assertEqual(result, [])

    @patch("src.watchlist.batch_runner.get_watchlist_path")
    def test_empty_json_returns_empty_list(self, mock_path):
        """空の JSON オブジェクトは空リストを返すこと"""
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write_watchlist(tmp, {})
            mock_path.return_value = p

            result = load_target_symbols()

        self.assertEqual(result, [])

    @patch("src.watchlist.batch_runner.get_watchlist_path")
    def test_multiple_symbols_in_market(self, mock_path):
        """1 マーケットに複数銘柄が含まれる場合、全件が生成されること"""
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write_watchlist(tmp, {"us": ["AAPL", "MSFT", "GOOG", "AMZN"]})
            mock_path.return_value = p

            result = load_target_symbols()

        self.assertEqual(len(result), 4)
        symbols = {t.symbol for t in result}
        self.assertIn("AAPL", symbols)
        self.assertIn("AMZN", symbols)

    @patch("src.watchlist.batch_runner.get_watchlist_path")
    def test_default_horizon_is_one(self, mock_path):
        """SymbolTask のデフォルト horizon は 1 であること"""
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write_watchlist(tmp, {"us": ["AAPL"]})
            mock_path.return_value = p

            result = load_target_symbols()

        self.assertEqual(result[0].horizon, 1)

    @patch("src.watchlist.batch_runner.load_index_membership_symbols_as_of")
    def test_as_of_date_loads_from_index_membership_history(self, mock_load_history):
        """as_of_date 指定時は index_membership_history を優先すること"""
        mock_load_history.return_value = [("us", "AAPL"), ("jp", "7203")]

        result = load_target_symbols(as_of_date="2025-01-31")

        self.assertEqual(len(result), 2)
        symbols = {(t.market, t.symbol) for t in result}
        self.assertIn(("us", "AAPL"), symbols)
        self.assertIn(("jp", "7203"), symbols)
        mock_load_history.assert_called_once_with("2025-01-31")

    @patch("src.watchlist.batch_runner.load_index_membership_symbols_as_of")
    @patch("src.watchlist.batch_runner.get_watchlist_path")
    def test_as_of_date_falls_back_to_watchlist_when_history_empty(
        self, mock_path, mock_load_history
    ):
        """履歴が空のとき watchlist.json にフォールバックすること"""
        mock_load_history.return_value = []
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write_watchlist(tmp, {"us": ["MSFT"]})
            mock_path.return_value = p

            result = load_target_symbols(as_of_date="2025-01-31")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].market, "us")
        self.assertEqual(result[0].symbol, "MSFT")


class TestRunParallel(unittest.TestCase):
    """run_parallel 関数のテスト"""

    def test_all_successful_tasks_collected(self):
        """全タスク成功時に全件の結果が収集されること"""
        tasks = [
            SymbolTask("us", "AAPL"),
            SymbolTask("us", "MSFT"),
            SymbolTask("jp", "7203"),
        ]

        def _success(task):
            return {"market": task.market, "symbol": task.symbol, "status": "success"}

        results = run_parallel(_success, tasks, max_workers=2)

        self.assertEqual(len(results), 3)
        statuses = {r["status"] for r in results}
        self.assertEqual(statuses, {"success"})

    def test_error_task_captured_with_status_error(self):
        """例外を投げるタスクが status=error として収集されること"""
        tasks = [SymbolTask("us", "BAD")]

        def _fail(task):
            raise ValueError("test error")

        results = run_parallel(_fail, tasks, max_workers=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "error")
        self.assertIn("error", results[0])

    def test_error_result_contains_market_symbol(self):
        """エラー時の結果に market / symbol が含まれること"""
        tasks = [SymbolTask("jp", "9984")]

        def _fail(task):
            raise RuntimeError("db error")

        results = run_parallel(_fail, tasks, max_workers=1)

        self.assertEqual(results[0]["market"], "jp")
        self.assertEqual(results[0]["symbol"], "9984")

    def test_empty_tasks_returns_empty_list(self):
        """タスクリストが空の場合は空リストが返ること"""
        results = run_parallel(lambda t: t, [], max_workers=2)
        self.assertEqual(results, [])

    def test_mixed_success_and_error(self):
        """成功とエラーが混在する場合、両方が収集されること"""

        def _mixed(task):
            if task.symbol == "BAD":
                raise ValueError("bad symbol")
            return {"market": task.market, "symbol": task.symbol, "status": "success"}

        tasks = [SymbolTask("us", "AAPL"), SymbolTask("us", "BAD")]
        results = run_parallel(_mixed, tasks, max_workers=2)

        self.assertEqual(len(results), 2)
        statuses = {r["status"] for r in results}
        self.assertIn("success", statuses)
        self.assertIn("error", statuses)

    def test_thread_pool_used_by_default(self):
        """デフォルト（use_process=False）は ThreadPoolExecutor が使われること"""
        tasks = [SymbolTask("us", "AAPL")]
        call_log = []

        def _track(task):
            call_log.append(task.symbol)
            return {"market": task.market, "symbol": task.symbol, "status": "success"}

        results = run_parallel(_track, tasks, use_process=False)

        self.assertEqual(len(results), 1)
        self.assertIn("AAPL", call_log)

    def test_result_count_matches_task_count(self):
        """タスク数と結果数が一致すること"""
        tasks = [SymbolTask("us", f"STOCK{i}") for i in range(5)]

        def _success(task):
            return {"market": task.market, "symbol": task.symbol, "status": "success"}

        results = run_parallel(_success, tasks, max_workers=3)
        self.assertEqual(len(results), 5)


class TestPrintSummary(unittest.TestCase):
    """print_summary 関数のテスト"""

    def test_runs_without_error_on_mixed_results(self):
        """成功/エラー/スキップ混在でエラーなく実行されること"""
        results = [
            {"status": "success", "market": "us", "symbol": "AAPL"},
            {"status": "error", "market": "us", "symbol": "BAD", "error": "test error"},
            {"status": "skip", "market": "jp", "symbol": "1234"},
        ]
        print_summary("テスト処理", results)

    def test_runs_without_error_on_all_success(self):
        """全成功でもエラーなく実行されること"""
        results = [
            {"status": "success", "market": "us", "symbol": "AAPL"},
            {"status": "success", "market": "jp", "symbol": "7203"},
        ]
        print_summary("全成功処理", results)

    def test_runs_without_error_on_empty_results(self):
        """空リストでもエラーなく実行されること"""
        print_summary("空処理", [])

    def test_runs_without_error_on_all_errors(self):
        """全エラーでもエラーなく実行されること"""
        results = [
            {"status": "error", "market": "us", "symbol": "BAD1", "error": "err1"},
            {"status": "error", "market": "us", "symbol": "BAD2", "error": "err2"},
        ]
        print_summary("全エラー処理", results)

    def test_missing_optional_keys_no_crash(self):
        """market / symbol / error キーが欠けていてもクラッシュしないこと"""
        results = [{"status": "error"}]
        print_summary("キー欠落処理", results)


if __name__ == "__main__":
    unittest.main()
