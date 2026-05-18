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
