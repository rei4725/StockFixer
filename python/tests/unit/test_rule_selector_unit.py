"""ユニットテスト: ルール評価エンジン（evaluate_all_symbols / _select_best_rule）"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any
from unittest.mock import MagicMock, patch


def _make_watchlist_file(market: str, symbols: list[str]) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8")
    json.dump({market: symbols}, f)
    f.close()
    return f.name


def _mock_result(rule: str, win_rate: float, net_profit: float) -> dict[str, Any]:
    return {
        "rule": rule,
        "win_rate": win_rate,
        "net_profit": net_profit,
        "description": f"{rule} desc",
        "total_return": 0.05,
        "num_trades": 10,
        "profit_factor": 1.5,
        "sharpe_ratio": 0.7,
        "max_drawdown": -0.05,
        "avg_win": 0.02,
        "avg_loss": 0.01,
        "buy_signals": 20,
        "market": "jp",
        "symbol": "7203",
    }


class TestSelectBestRule(unittest.TestCase):
    def test_returns_none_when_no_valid_rules(self):
        from src.backtest.rule_selector import _select_best_rule

        results = [_mock_result("r1", 0.3, -1000)]  # win_rate < 0.5
        self.assertIsNone(_select_best_rule(results))

    def test_returns_none_when_net_profit_zero(self):
        from src.backtest.rule_selector import _select_best_rule

        results = [_mock_result("r1", 0.6, 0.0)]  # net_profit not > 0
        self.assertIsNone(_select_best_rule(results))

    def test_returns_best_by_win_rate(self):
        from src.backtest.rule_selector import _select_best_rule

        results = [
            _mock_result("r1", 0.6, 1000),
            _mock_result("r2", 0.7, 500),
        ]
        best = _select_best_rule(results)
        self.assertIsNotNone(best)
        self.assertEqual(best["rule"], "r2")

    def test_tiebreak_by_net_profit(self):
        from src.backtest.rule_selector import _select_best_rule

        results = [
            _mock_result("r1", 0.6, 1000),
            _mock_result("r2", 0.6, 2000),
        ]
        best = _select_best_rule(results)
        self.assertEqual(best["rule"], "r2")

    def test_empty_results_returns_none(self):
        from src.backtest.rule_selector import _select_best_rule

        self.assertIsNone(_select_best_rule([]))


class TestEvaluateAllSymbols(unittest.TestCase):
    def setUp(self):
        self.watchlist_path = _make_watchlist_file("jp", ["7203", "6758"])

    def tearDown(self):
        os.unlink(self.watchlist_path)

    def test_empty_watchlist_returns_empty_summary(self):
        from src.backtest.rule_selector import evaluate_all_symbols

        empty_wl = _make_watchlist_file("jp", [])
        try:
            with patch("src.backtest.rule_selector._WATCHLIST_PATH", empty_wl):
                result = evaluate_all_symbols("jp", "2024-01-01", "2024-12-31")
        finally:
            os.unlink(empty_wl)
        self.assertIn("evaluated", result)
        self.assertEqual(result["evaluated"], 0)

    def test_no_valid_rule_marks_symbol_as_no_rule(self):
        from src.backtest.rule_selector import evaluate_all_symbols

        with (
            patch("src.backtest.rule_selector._WATCHLIST_PATH", self.watchlist_path),
            patch(
                "src.backtest.rule_selector.run_rule_backtest",
                return_value=[_mock_result("r1", 0.3, -100)],  # 有効ルールなし
            ),
            patch("src.backtest.rule_selector.upsert_rule_best"),
            patch("src.backtest.rule_selector.load_rule_best_all", return_value=MagicMock()),
        ):
            result = evaluate_all_symbols("jp", "2024-01-01", "2024-12-31")
        self.assertIn("effective", result)
        self.assertEqual(result["effective"], 0)

    def test_valid_rule_is_saved(self):
        from src.backtest.rule_selector import evaluate_all_symbols

        with (
            patch("src.backtest.rule_selector._WATCHLIST_PATH", self.watchlist_path),
            patch(
                "src.backtest.rule_selector.run_rule_backtest",
                return_value=[_mock_result("rsi_contrarian", 0.65, 50000)],
            ),
            patch("src.backtest.rule_selector.upsert_rule_best") as mock_upsert,
            patch("src.backtest.rule_selector.load_rule_best_all", return_value=MagicMock()),
        ):
            result = evaluate_all_symbols("jp", "2024-01-01", "2024-12-31")
        self.assertGreater(mock_upsert.call_count, 0)
        self.assertGreater(result["effective"], 0)

    def test_exception_per_symbol_is_handled(self):
        from src.backtest.rule_selector import evaluate_all_symbols

        with (
            patch("src.backtest.rule_selector._WATCHLIST_PATH", self.watchlist_path),
            patch(
                "src.backtest.rule_selector.run_rule_backtest",
                side_effect=RuntimeError("backtest error"),
            ),
            patch("src.backtest.rule_selector.upsert_rule_best"),
            patch("src.backtest.rule_selector.load_rule_best_all", return_value=MagicMock()),
        ):
            result = evaluate_all_symbols("jp", "2024-01-01", "2024-12-31")
        # 例外が伝播せず結果が返る
        self.assertIsInstance(result, dict)

    def test_exception_per_symbol_counts_as_error_not_skipped(self):
        """評価の失敗は errors に数え、有効ルールなしの skipped と混ぜない（#1087）。"""
        from src.backtest.rule_selector import evaluate_all_symbols

        with (
            patch("src.backtest.rule_selector._WATCHLIST_PATH", self.watchlist_path),
            patch(
                "src.backtest.rule_selector.run_rule_backtest",
                side_effect=RuntimeError("backtest error"),
            ),
            patch("src.backtest.rule_selector.upsert_rule_best"),
        ):
            result = evaluate_all_symbols("jp", "2024-01-01", "2024-12-31")
        self.assertEqual(result["errors"], 2)
        self.assertEqual(result["skipped"], 0)

    def test_upsert_failure_counts_as_error(self):
        from src.backtest.rule_selector import evaluate_all_symbols

        with (
            patch("src.backtest.rule_selector._WATCHLIST_PATH", self.watchlist_path),
            patch(
                "src.backtest.rule_selector.run_rule_backtest",
                return_value=[_mock_result("rsi_contrarian", 0.65, 50000)],
            ),
            patch(
                "src.backtest.rule_selector.upsert_rule_best",
                side_effect=RuntimeError("db error"),
            ),
        ):
            result = evaluate_all_symbols("jp", "2024-01-01", "2024-12-31")
        self.assertEqual(result["errors"], 2)
        self.assertEqual(result["skipped"], 0)
        self.assertEqual(result["effective"], 0)

    def test_no_valid_rule_counts_as_skipped_not_error(self):
        from src.backtest.rule_selector import evaluate_all_symbols

        with (
            patch("src.backtest.rule_selector._WATCHLIST_PATH", self.watchlist_path),
            patch(
                "src.backtest.rule_selector.run_rule_backtest",
                return_value=[_mock_result("r1", 0.3, -100)],
            ),
            patch("src.backtest.rule_selector.upsert_rule_best"),
        ):
            result = evaluate_all_symbols("jp", "2024-01-01", "2024-12-31")
        self.assertEqual(result["skipped"], 2)
        self.assertEqual(result["errors"], 0)

    def test_fetch_failure_counts_as_error_not_skipped(self):
        """値段の取得の失敗（空）は errors に数え、有効ルールなしの skipped と混ぜない（#1270）。"""
        import pandas as pd

        from src.backtest.rule_selector import evaluate_all_symbols

        port = MagicMock()
        # 7203 は取得が空、6758 は取れるがルールが無いので有効ルールなし
        port.download.side_effect = [pd.DataFrame(), pd.DataFrame({"Close": [100.0, 101.0]})]
        port.add_technical_indicators.side_effect = lambda df: df
        with (
            patch("src.backtest.rule_selector._WATCHLIST_PATH", self.watchlist_path),
            patch("src.backtest.rule_selector.ALL_RULES", []),
            patch("src.backtest.rule_backtester.get_backtest_data_port", return_value=port),
            patch("src.backtest.rule_selector.upsert_rule_best") as mock_upsert,
        ):
            result = evaluate_all_symbols("jp", "2024-01-01", "2024-12-31")
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["evaluated"], 1)
        mock_upsert.assert_not_called()

    def _seed_previous_week(self) -> None:
        from src.utils.db.rule_results import upsert_rule_best

        for symbol in ("7203", "6758"):
            upsert_rule_best(
                market="jp",
                symbol=symbol,
                best_rule="golden_cross",
                win_rate=0.7,
                net_profit=10000.0,
                num_trades=10,
                profit_factor=1.5,
                max_drawdown=-0.05,
                backtest_start="2023-01-01",
                backtest_end="2023-12-31",
            )

    def test_no_valid_rule_drops_previous_effective_row(self):
        """今週の評価で有効ルールが無い銘柄は、前の週の有効の行を残さない（#1275）。"""
        from src.backtest.rule_selector import evaluate_all_symbols
        from src.utils.db.rule_results import load_effective_rules

        self._seed_previous_week()
        with (
            patch("src.backtest.rule_selector._WATCHLIST_PATH", self.watchlist_path),
            patch(
                "src.backtest.rule_selector.run_rule_backtest",
                return_value=[_mock_result("golden_cross", 0.0, -5000)],  # 全敗
            ),
        ):
            result = evaluate_all_symbols("jp", "2024-01-01", "2024-12-31")
        symbols = set(load_effective_rules("jp")["symbol"])
        self.assertNotIn("7203", symbols)
        self.assertNotIn("6758", symbols)
        self.assertEqual(result["skipped"], 2)
        self.assertEqual(result["errors"], 0)

    def test_valid_rule_replaces_previous_row(self):
        from src.backtest.rule_selector import evaluate_all_symbols
        from src.utils.db.rule_results import load_effective_rules

        self._seed_previous_week()
        with (
            patch("src.backtest.rule_selector._WATCHLIST_PATH", self.watchlist_path),
            patch(
                "src.backtest.rule_selector.run_rule_backtest",
                return_value=[_mock_result("rsi_contrarian", 0.65, 50000)],
            ),
        ):
            evaluate_all_symbols("jp", "2024-01-01", "2024-12-31")
        df = load_effective_rules("jp")
        rows = df[df["symbol"].isin(["7203", "6758"])]
        self.assertEqual(len(rows), 2)
        self.assertEqual(set(rows["best_rule"]), {"rsi_contrarian"})

    def test_evaluation_error_keeps_previous_row(self):
        from src.backtest.rule_selector import evaluate_all_symbols
        from src.utils.db.rule_results import load_effective_rules

        self._seed_previous_week()
        with (
            patch("src.backtest.rule_selector._WATCHLIST_PATH", self.watchlist_path),
            patch(
                "src.backtest.rule_selector.run_rule_backtest",
                side_effect=RuntimeError("backtest error"),
            ),
        ):
            result = evaluate_all_symbols("jp", "2024-01-01", "2024-12-31")
        symbols = set(load_effective_rules("jp")["symbol"])
        self.assertIn("7203", symbols)
        self.assertIn("6758", symbols)
        self.assertEqual(result["errors"], 2)

    def test_fetch_failure_keeps_previous_row(self):
        """取得に失敗した銘柄の行は残し、取得できて有効ルールが無い銘柄の行は消す。"""
        import pandas as pd

        from src.backtest.rule_selector import evaluate_all_symbols
        from src.utils.db.rule_results import load_effective_rules

        self._seed_previous_week()
        port = MagicMock()
        # 7203 は取得が空、6758 は取れるがルールが無いので有効ルールなし
        port.download.side_effect = [pd.DataFrame(), pd.DataFrame({"Close": [100.0, 101.0]})]
        port.add_technical_indicators.side_effect = lambda df: df
        with (
            patch("src.backtest.rule_selector._WATCHLIST_PATH", self.watchlist_path),
            patch("src.backtest.rule_selector.ALL_RULES", []),
            patch("src.backtest.rule_backtester.get_backtest_data_port", return_value=port),
        ):
            result = evaluate_all_symbols("jp", "2024-01-01", "2024-12-31")
        symbols = set(load_effective_rules("jp")["symbol"])
        self.assertIn("7203", symbols)
        self.assertNotIn("6758", symbols)
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["skipped"], 1)

    def _seed_off_watchlist(self, symbol: str = "9999") -> None:
        from src.utils.db.rule_results import upsert_rule_best

        upsert_rule_best(
            market="jp",
            symbol=symbol,
            best_rule="golden_cross",
            win_rate=0.7,
            net_profit=10000.0,
            num_trades=10,
            profit_factor=1.5,
            max_drawdown=-0.05,
            backtest_start="2023-01-01",
            backtest_end="2023-12-31",
        )

    def test_off_watchlist_row_is_dropped(self):
        """ウォッチリストから外れた銘柄の行は、全体の評価のあと残さない（#1294）。"""
        from src.backtest.rule_selector import evaluate_all_symbols
        from src.utils.db.rule_results import load_effective_rules

        self._seed_off_watchlist()
        with (
            patch("src.backtest.rule_selector._WATCHLIST_PATH", self.watchlist_path),
            patch(
                "src.backtest.rule_selector.run_rule_backtest",
                return_value=[_mock_result("rsi_contrarian", 0.65, 50000)],
            ),
        ):
            result = evaluate_all_symbols("jp", "2024-01-01", "2024-12-31")
        symbols = set(load_effective_rules("jp")["symbol"])
        self.assertNotIn("9999", symbols)
        self.assertIn("7203", symbols)
        self.assertIn("6758", symbols)
        self.assertEqual(result["errors"], 0)

    def test_off_watchlist_row_kept_when_watchlist_empty(self):
        from src.backtest.rule_selector import evaluate_all_symbols
        from src.utils.db.rule_results import load_effective_rules

        self._seed_off_watchlist()
        empty_wl = _make_watchlist_file("jp", [])
        try:
            with patch("src.backtest.rule_selector._WATCHLIST_PATH", empty_wl):
                evaluate_all_symbols("jp", "2024-01-01", "2024-12-31")
        finally:
            os.unlink(empty_wl)
        self.assertIn("9999", set(load_effective_rules("jp")["symbol"]))

    def test_off_watchlist_row_kept_when_watchlist_unreadable(self):
        from src.backtest.rule_selector import evaluate_all_symbols
        from src.utils.db.rule_results import load_effective_rules

        self._seed_off_watchlist()
        missing = self.watchlist_path + ".missing"
        with patch("src.backtest.rule_selector._WATCHLIST_PATH", missing):
            with self.assertRaises(OSError):
                evaluate_all_symbols("jp", "2024-01-01", "2024-12-31")
        self.assertIn("9999", set(load_effective_rules("jp")["symbol"]))

    def test_off_watchlist_row_kept_when_symbols_given(self):
        """手で銘柄を渡した評価では、ほかの銘柄の行を消さない。"""
        from src.backtest.rule_selector import evaluate_all_symbols
        from src.utils.db.rule_results import load_effective_rules

        self._seed_off_watchlist()
        with (
            patch("src.backtest.rule_selector._WATCHLIST_PATH", self.watchlist_path),
            patch(
                "src.backtest.rule_selector.run_rule_backtest",
                return_value=[_mock_result("rsi_contrarian", 0.65, 50000)],
            ),
        ):
            evaluate_all_symbols("jp", "2024-01-01", "2024-12-31", symbols=["7203"])
        symbols = set(load_effective_rules("jp")["symbol"])
        self.assertIn("9999", symbols)
        self.assertIn("7203", symbols)

    def test_off_watchlist_prune_failure_counts_as_error(self):
        from src.backtest.rule_selector import evaluate_all_symbols

        with (
            patch("src.backtest.rule_selector._WATCHLIST_PATH", self.watchlist_path),
            patch(
                "src.backtest.rule_selector.run_rule_backtest",
                return_value=[_mock_result("rsi_contrarian", 0.65, 50000)],
            ),
            patch("src.backtest.rule_selector.upsert_rule_best"),
            patch(
                "src.backtest.rule_selector.delete_rule_best_except",
                side_effect=RuntimeError("db error"),
            ),
        ):
            result = evaluate_all_symbols("jp", "2024-01-01", "2024-12-31")
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["effective"], 2)

    def test_returns_summary_keys(self):
        from src.backtest.rule_selector import evaluate_all_symbols

        with (
            patch("src.backtest.rule_selector._WATCHLIST_PATH", self.watchlist_path),
            patch("src.backtest.rule_selector.run_rule_backtest", return_value=[]),
            patch("src.backtest.rule_selector.upsert_rule_best"),
            patch("src.backtest.rule_selector.load_rule_best_all", return_value=MagicMock()),
        ):
            result = evaluate_all_symbols("jp", "2024-01-01", "2024-12-31")
        for key in ("evaluated", "effective", "skipped"):
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()
