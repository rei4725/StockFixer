"""ユニットテスト: ルールベース日次シグナルパイプライン（rule_engine BC）"""

from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


def _make_ohlcv(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.linspace(1000, 1100, n)
    return pd.DataFrame(
        {
            "Open": close - 5,
            "High": close + 10,
            "Low": close - 10,
            "Close": close,
            "Volume": np.full(n, 500_000, dtype=float),
        },
        index=idx,
    )


class TestGetTodaySignal(unittest.TestCase):
    def test_unknown_rule_returns_hold(self):
        from src.rule_engine.pipeline import _get_today_signal

        signal, price = _get_today_signal("jp", "7203", "unknown_rule")
        self.assertEqual(signal, 0)
        self.assertIsNone(price)

    def test_empty_df_returns_hold(self):
        from src.rule_engine.pipeline import _get_today_signal

        with (
            patch("src.rule_engine.pipeline.get_ticker", return_value="7203.T"),
            patch(
                "src.rule_engine.pipeline.yf_client.download",
                return_value=pd.DataFrame(),
            ),
        ):
            signal, price = _get_today_signal("jp", "7203", "rsi_contrarian")
        self.assertEqual(signal, 0)
        self.assertIsNone(price)

    def test_valid_df_returns_signal_and_price(self):
        from src.rule_engine.pipeline import _get_today_signal

        df = _make_ohlcv()
        with (
            patch("src.rule_engine.pipeline.get_ticker", return_value="7203.T"),
            patch("src.rule_engine.pipeline.yf_client.download", return_value=df),
            patch("src.rule_engine.pipeline.add_technical_indicators", return_value=df),
        ):
            signal, price = _get_today_signal("jp", "7203", "rsi_contrarian")
        self.assertIn(signal, [-1, 0, 1])
        self.assertIsInstance(price, float)


class TestRunRuleSignalPipeline(unittest.TestCase):
    def test_empty_effective_rules_returns_empty(self):
        from src.rule_engine.pipeline import run_rule_signal_pipeline

        with patch(
            "src.rule_engine.pipeline.load_effective_rules",
            return_value=pd.DataFrame(),
        ):
            result = run_rule_signal_pipeline("jp")
        self.assertEqual(result, [])

    def test_returns_results_for_each_symbol(self):
        from src.rule_engine.pipeline import run_rule_signal_pipeline

        effective_df = pd.DataFrame(
            [
                {
                    "symbol": "7203",
                    "best_rule": "rsi_contrarian",
                    "win_rate": 0.6,
                    "net_profit": 1000,
                },
                {
                    "symbol": "6758",
                    "best_rule": "ema_momentum",
                    "win_rate": 0.55,
                    "net_profit": 500,
                },
            ]
        )
        df = _make_ohlcv()
        with (
            patch(
                "src.rule_engine.pipeline.load_effective_rules",
                return_value=effective_df,
            ),
            patch("src.rule_engine.pipeline.get_ticker", return_value="TICKER"),
            patch("src.rule_engine.pipeline.yf_client.download", return_value=df),
            patch("src.rule_engine.pipeline.add_technical_indicators", return_value=df),
            patch("src.rule_engine.pipeline.upsert_rule_signal"),
        ):
            result = run_rule_signal_pipeline("jp")
        self.assertEqual(len(result), 2)
        self.assertIn("symbol", result[0])
        self.assertIn("signal_label", result[0])

    def test_signal_exception_is_caught(self):
        from src.rule_engine.pipeline import run_rule_signal_pipeline

        effective_df = pd.DataFrame(
            [{"symbol": "7203", "best_rule": "rsi_contrarian", "win_rate": 0.6, "net_profit": 100}]
        )
        with (
            patch(
                "src.rule_engine.pipeline.load_effective_rules",
                return_value=effective_df,
            ),
            patch(
                "src.rule_engine.pipeline._get_today_signal",
                side_effect=RuntimeError("fetch error"),
            ),
        ):
            result = run_rule_signal_pipeline("jp")
        self.assertEqual(result, [])

    def test_signal_date_defaults_to_today(self):
        from src.rule_engine.pipeline import run_rule_signal_pipeline

        with patch(
            "src.rule_engine.pipeline.load_effective_rules",
            return_value=pd.DataFrame(),
        ):
            result = run_rule_signal_pipeline("jp", signal_date=None)
        self.assertEqual(result, [])

    def test_custom_signal_date(self):
        from src.rule_engine.pipeline import run_rule_signal_pipeline

        with patch(
            "src.rule_engine.pipeline.load_effective_rules",
            return_value=pd.DataFrame(),
        ):
            result = run_rule_signal_pipeline("jp", signal_date=date(2024, 1, 15))
        self.assertEqual(result, [])


class TestExecuteRulePaperTrades(unittest.TestCase):
    def test_empty_signals_returns_zero_counts(self):
        from src.trading.rule_execution import execute_rule_paper_trades

        mock_broker = MagicMock()
        mock_broker.get_positions.return_value = []

        with patch("src.trading.rule_execution.PaperBroker", return_value=mock_broker):
            result = execute_rule_paper_trades([], "jp")
        self.assertEqual(result["buy_orders"], 0)
        self.assertEqual(result["sell_orders"], 0)

    def test_buy_signal_places_order(self):
        from src.trading.rule_execution import execute_rule_paper_trades

        mock_broker = MagicMock()
        mock_broker.get_positions.return_value = []

        signals = [{"symbol": "7203", "signal": 1, "price": 1000.0}]
        with (
            patch("src.trading.rule_execution.PaperBroker", return_value=mock_broker),
            patch("src.trading.rule_execution.get_ticker", return_value="7203.T"),
        ):
            result = execute_rule_paper_trades(signals, "jp")
        self.assertEqual(result["buy_orders"], 1)

    def test_sell_signal_places_order_when_holding(self):
        from src.trading.rule_execution import execute_rule_paper_trades

        mock_broker = MagicMock()
        mock_broker.get_positions.return_value = [{"symbol": "7203", "qty": 100}]

        signals = [{"symbol": "7203", "signal": -1, "price": 1100.0}]
        with (
            patch("src.trading.rule_execution.PaperBroker", return_value=mock_broker),
            patch("src.trading.rule_execution.get_ticker", return_value="7203.T"),
        ):
            result = execute_rule_paper_trades(signals, "jp")
        self.assertEqual(result["sell_orders"], 1)

    def test_hold_signal_is_skipped(self):
        from src.trading.rule_execution import execute_rule_paper_trades

        mock_broker = MagicMock()
        mock_broker.get_positions.return_value = []

        signals = [{"symbol": "7203", "signal": 0, "price": 1000.0}]
        with (
            patch("src.trading.rule_execution.PaperBroker", return_value=mock_broker),
        ):
            result = execute_rule_paper_trades(signals, "jp")
        self.assertEqual(result["skipped"], 1)

    def test_broker_exception_is_caught(self):
        from src.trading.rule_execution import execute_rule_paper_trades

        mock_broker = MagicMock()
        mock_broker.get_positions.side_effect = RuntimeError("broker error")

        signals = [{"symbol": "7203", "signal": 1, "price": 1000.0}]
        with (
            patch("src.trading.rule_execution.PaperBroker", return_value=mock_broker),
            patch("src.trading.rule_execution.get_ticker", return_value="7203.T"),
        ):
            result = execute_rule_paper_trades(signals, "jp")
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
