"""batch_runner モジュールのユニットテスト"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from src.watchlist.batch_runner import load_target_symbols, print_summary, run_parallel
from src.domain.types import SymbolTask
from src.watchlist.types import BatchFailure, BatchResult


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
    """run_parallel 関数のテスト（BatchResult 返却）"""

    def test_returns_batch_result(self):
        """run_parallel が BatchResult を返すこと"""
        tasks = [SymbolTask("us", "AAPL")]

        def _success(task):
            return {"market": task.market, "symbol": task.symbol, "status": "success"}

        result = run_parallel(_success, tasks, max_workers=1)
        self.assertIsInstance(result, BatchResult)

    def test_all_successful_tasks_collected(self):
        """全タスク成功時に succeeded に全件が収集されること"""
        tasks = [
            SymbolTask("us", "AAPL"),
            SymbolTask("us", "MSFT"),
            SymbolTask("jp", "7203"),
        ]

        def _success(task):
            return {"market": task.market, "symbol": task.symbol, "status": "success"}

        result = run_parallel(_success, tasks, max_workers=2)

        self.assertEqual(len(result.succeeded), 3)
        self.assertEqual(len(result.failed), 0)
        self.assertEqual(len(result.skipped), 0)

    def test_error_task_captured_in_failed(self):
        """例外を投げるタスクが failed に BatchFailure として収集されること"""
        tasks = [SymbolTask("us", "BAD")]

        def _fail(task):
            raise ValueError("test error")

        result = run_parallel(_fail, tasks, max_workers=1)

        self.assertEqual(len(result.succeeded), 0)
        self.assertEqual(len(result.failed), 1)
        self.assertIsInstance(result.failed[0], BatchFailure)

    def test_failed_contains_market_symbol_error(self):
        """BatchFailure に market / symbol / error が設定されること"""
        tasks = [SymbolTask("jp", "9984")]

        def _fail(task):
            raise RuntimeError("db error")

        result = run_parallel(_fail, tasks, max_workers=1)

        bf = result.failed[0]
        self.assertEqual(bf.market, "jp")
        self.assertEqual(bf.symbol, "9984")
        self.assertIn("db error", bf.error)

    def test_empty_tasks_returns_empty_batch_result(self):
        """タスクリストが空の場合は空の BatchResult が返ること"""
        result = run_parallel(lambda t: t, [], max_workers=2)
        self.assertIsInstance(result, BatchResult)
        self.assertEqual(len(result.succeeded), 0)
        self.assertEqual(len(result.failed), 0)

    def test_skip_status_goes_to_skipped(self):
        """status=="skip" を返すタスクが skipped に収集されること"""
        tasks = [SymbolTask("us", "SKIP")]

        def _skip(task):
            return {"market": task.market, "symbol": task.symbol, "status": "skip"}

        result = run_parallel(_skip, tasks, max_workers=1)

        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(len(result.succeeded), 0)
        self.assertEqual(len(result.failed), 0)

    def test_error_status_return_goes_to_failed(self):
        """関数が status=="error" を返した場合も failed に収集されること"""
        tasks = [SymbolTask("us", "ERR")]

        def _err(task):
            return {
                "market": task.market,
                "symbol": task.symbol,
                "status": "error",
                "error": "内部エラー",
            }

        result = run_parallel(_err, tasks, max_workers=1)

        self.assertEqual(len(result.failed), 1)
        self.assertEqual(len(result.succeeded), 0)
        bf = result.failed[0]
        self.assertEqual(bf.market, "us")
        self.assertEqual(bf.symbol, "ERR")

    def test_mixed_results_distributed_correctly(self):
        """成功・エラー・スキップが正しく分類されること"""

        def _mixed(task):
            if task.symbol == "BAD":
                raise ValueError("bad symbol")
            if task.symbol == "SKIP":
                return {"market": task.market, "symbol": task.symbol, "status": "skip"}
            return {"market": task.market, "symbol": task.symbol, "status": "success"}

        tasks = [
            SymbolTask("us", "AAPL"),
            SymbolTask("us", "BAD"),
            SymbolTask("us", "SKIP"),
        ]
        result = run_parallel(_mixed, tasks, max_workers=2)

        self.assertEqual(len(result.succeeded), 1)
        self.assertEqual(len(result.failed), 1)
        self.assertEqual(len(result.skipped), 1)

    def test_thread_pool_used_by_default(self):
        """デフォルト（use_process=False）で BatchResult が返ること"""
        tasks = [SymbolTask("us", "AAPL")]

        def _track(task):
            return {"market": task.market, "symbol": task.symbol, "status": "success"}

        result = run_parallel(_track, tasks, use_process=False)
        self.assertEqual(len(result.succeeded), 1)


class TestPrintSummary(unittest.TestCase):
    """print_summary 関数のテスト（BatchResult ベース）"""

    def _make_batch(self, succeeded=None, failed=None, skipped=None) -> BatchResult:
        return BatchResult(
            succeeded=succeeded or [],
            failed=failed or [],
            skipped=skipped or [],
        )

    @patch("src.watchlist.batch_runner.send_webhook_notification", create=True)
    def test_runs_without_error_on_mixed_results(self, _mock_notify):
        """成功/エラー/スキップ混在でエラーなく実行されること"""
        batch = self._make_batch(
            succeeded=[{"market": "us", "symbol": "AAPL", "status": "success"}],
            failed=[BatchFailure(market="us", symbol="BAD", error="test error")],
            skipped=[{"market": "jp", "symbol": "1234", "status": "skip"}],
        )
        print_summary("テスト処理", batch)

    def test_runs_without_error_on_all_success(self):
        """全成功でもエラーなく実行されること"""
        batch = self._make_batch(
            succeeded=[
                {"market": "us", "symbol": "AAPL", "status": "success"},
                {"market": "jp", "symbol": "7203", "status": "success"},
            ],
        )
        print_summary("全成功処理", batch)

    def test_runs_without_error_on_empty_results(self):
        """空の BatchResult でもエラーなく実行されること"""
        print_summary("空処理", self._make_batch())

    @patch("src.watchlist.batch_runner.send_webhook_notification", create=True)
    def test_runs_without_error_on_all_errors(self, _mock_notify):
        """全エラーでもエラーなく実行されること"""
        batch = self._make_batch(
            failed=[
                BatchFailure(market="us", symbol="BAD1", error="err1"),
                BatchFailure(market="us", symbol="BAD2", error="err2"),
            ],
        )
        print_summary("全エラー処理", batch)

    @patch("src.reporting.discord.discord_utils.send_webhook_notification")
    def test_discord_alert_sent_when_failures_exist(self, mock_notify):
        """failed が存在するとき Discord 通知が試みられること"""
        batch = self._make_batch(
            failed=[BatchFailure(market="us", symbol="BAD", error="fail")],
        )
        print_summary("通知テスト", batch)
        mock_notify.assert_called_once()

    def test_no_discord_alert_when_no_failures(self):
        """failed が空のとき Discord 通知が呼ばれないこと"""
        batch = self._make_batch(
            succeeded=[{"market": "us", "symbol": "AAPL", "status": "success"}],
        )
        with patch("src.reporting.discord.discord_utils.send_webhook_notification") as mock_notify:
            print_summary("成功のみ", batch)
        mock_notify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
