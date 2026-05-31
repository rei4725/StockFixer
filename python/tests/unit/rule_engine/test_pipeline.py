"""ユニットテスト: rule_engine BC パイプライン"""

from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


class _MockMarketDataPort:
    def __init__(self, df):
        self._df = df

    def get_ohlcv_with_indicators(self, ticker, lookback_days):
        return self._df


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


class TestRuleEngineTypes(unittest.TestCase):
    def test_rule_signal_instantiation(self):
        from src.rule_engine.types import RuleSignal

        sig = RuleSignal(
            symbol="7203",
            rule="rsi_contrarian",
            signal=1,
            signal_label="BUY",
            price=1000.0,
            win_rate=0.6,
            net_profit=1000.0,
        )
        self.assertEqual(sig.symbol, "7203")
        self.assertEqual(sig.signal, 1)

    def test_rule_result_instantiation(self):
        from src.rule_engine.types import RuleResult, RuleSignal

        sig = RuleSignal("7203", "rsi_contrarian", 1, "BUY", 1000.0, 0.6, 1000.0)
        result = RuleResult(market="jp", signal_date=date(2024, 1, 15), signals=[sig])
        self.assertEqual(result.market, "jp")
        self.assertEqual(len(result.signals), 1)


class TestRuleEngineRules(unittest.TestCase):
    def test_trading_rule_protocol_satisfied(self):
        from src.rule_engine.ports import TradingRule
        from src.rule_engine.rules.technical import RSIContrarianRule

        rule = RSIContrarianRule()
        self.assertIsInstance(rule, TradingRule)

    def test_all_rules_generate_signal(self):
        from src.rule_engine.rules import ALL_RULES

        df = _make_ohlcv()
        for rule in ALL_RULES:
            sig = rule.generate_signal(df)
            self.assertEqual(len(sig), len(df))
            self.assertTrue(sig.isin([-1, 0, 1]).all())


class TestRunRuleSignalPipeline(unittest.TestCase):
    def test_empty_effective_rules_returns_empty(self):
        from src.rule_engine.pipeline import run_rule_signal_pipeline

        with patch(
            "src.rule_engine.pipeline.load_effective_rules",
            return_value=pd.DataFrame(),
        ):
            result = run_rule_signal_pipeline("jp", market_data_port=MagicMock())
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
        mock_port = _MockMarketDataPort(df)
        with (
            patch("src.rule_engine.pipeline.load_effective_rules", return_value=effective_df),
            patch("src.rule_engine.pipeline.upsert_rule_signal"),
        ):
            result = run_rule_signal_pipeline("jp", market_data_port=mock_port)
        self.assertEqual(len(result), 2)
        self.assertIn("signal_label", result[0])

    def test_signal_date_defaults_to_today(self):
        from src.rule_engine.pipeline import run_rule_signal_pipeline

        with patch(
            "src.rule_engine.pipeline.load_effective_rules",
            return_value=pd.DataFrame(),
        ):
            result = run_rule_signal_pipeline("jp", signal_date=None, market_data_port=MagicMock())
        self.assertEqual(result, [])

    def test_no_backtest_or_trading_import(self):
        """rule_engine.pipeline が backtest/trading に依存しないことを確認"""
        import sys

        # モジュールをリロードして依存を確認
        mods_before = set(sys.modules.keys())
        import src.rule_engine.pipeline  # noqa: F401

        mods_after = set(sys.modules.keys())
        new_mods = mods_after - mods_before
        for mod in new_mods:
            self.assertFalse(
                mod.startswith("src.backtest"),
                f"rule_engine.pipeline が src.backtest に依存: {mod}",
            )
            self.assertFalse(
                mod.startswith("src.trading"),
                f"rule_engine.pipeline が src.trading に依存: {mod}",
            )
            self.assertFalse(
                mod.startswith("src.prediction"),
                f"rule_engine.pipeline が src.prediction に依存: {mod}",
            )


if __name__ == "__main__":
    unittest.main()
