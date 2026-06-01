"""ユニットテスト: src/backtest/rules/composite.py"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.backtest.rules.composite import AndRule, OrRule


def _make_df(n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"Close": np.linspace(100, 110, n)}, index=idx)


class _ConstRule:
    """テスト用: 定数シグナルを返すスタブルール"""

    def __init__(self, signal: int, name: str = "stub"):
        self._signal = signal
        self.name = name
        self.description = f"const_{signal}"

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self._signal, index=df.index)


class _SeriesRule:
    """テスト用: 列ごとに異なるシグナルを返すスタブルール"""

    def __init__(self, signals: list[int], name: str = "series"):
        self._signals = signals
        self.name = name
        self.description = "series"

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self._signals, index=df.index)


class TestAndRuleBuy(unittest.TestCase):
    def test_all_buy_returns_buy(self):
        df = _make_df()
        rule = AndRule([_ConstRule(1, "a"), _ConstRule(1, "b")])
        sig = rule.generate_signal(df)
        self.assertTrue((sig == 1).all())

    def test_mixed_buy_neutral_returns_neutral(self):
        df = _make_df()
        rule = AndRule([_ConstRule(1, "a"), _ConstRule(0, "b")])
        sig = rule.generate_signal(df)
        self.assertTrue((sig == 0).all())

    def test_any_sell_overrides_to_sell(self):
        df = _make_df()
        rule = AndRule([_ConstRule(1, "a"), _ConstRule(-1, "b")])
        sig = rule.generate_signal(df)
        self.assertTrue((sig == -1).all())

    def test_all_sell_returns_sell(self):
        df = _make_df()
        rule = AndRule([_ConstRule(-1, "a"), _ConstRule(-1, "b")])
        sig = rule.generate_signal(df)
        self.assertTrue((sig == -1).all())

    def test_three_rules_all_buy(self):
        df = _make_df()
        rule = AndRule([_ConstRule(1, "a"), _ConstRule(1, "b"), _ConstRule(1, "c")])
        sig = rule.generate_signal(df)
        self.assertTrue((sig == 1).all())

    def test_sell_takes_priority_over_buy(self):
        """sell シグナルは buy より優先される（sell_any が True なら -1 を上書き）"""
        df = _make_df(3)
        rule = AndRule([_SeriesRule([1, 1, 1], "buy_rule"), _SeriesRule([0, -1, 1], "sell_rule")])
        sig = rule.generate_signal(df)
        # row0: buy_rule=1, sell_rule=0 → buy_all=False(1&(0==1)), sell_any=False → 0
        self.assertEqual(sig.iloc[0], 0)
        # row1: buy_rule=1, sell_rule=-1 → sell_any=True → -1
        self.assertEqual(sig.iloc[1], -1)
        # row2: buy_rule=1, sell_rule=1 → buy_all=True, sell_any=False → 1
        self.assertEqual(sig.iloc[2], 1)


class TestAndRuleNaming(unittest.TestCase):
    def test_auto_name(self):
        rule = AndRule([_ConstRule(1, "rsi"), _ConstRule(1, "macd")])
        self.assertIn("rsi", rule.name)
        self.assertIn("macd", rule.name)

    def test_custom_name(self):
        rule = AndRule([_ConstRule(1, "a")], name="my_and_rule")
        self.assertEqual(rule.name, "my_and_rule")

    def test_description_contains_child_descriptions(self):
        rule = AndRule([_ConstRule(1, "a"), _ConstRule(1, "b")])
        self.assertIn("const_1", rule.description)


class TestOrRuleBuy(unittest.TestCase):
    def test_one_of_two_buy_returns_buy_default_threshold(self):
        df = _make_df()
        rule = OrRule([_ConstRule(1, "a"), _ConstRule(0, "b")], threshold=0.5)
        sig = rule.generate_signal(df)
        self.assertTrue((sig == 1).all())

    def test_all_neutral_returns_neutral(self):
        df = _make_df()
        rule = OrRule([_ConstRule(0, "a"), _ConstRule(0, "b")])
        sig = rule.generate_signal(df)
        self.assertTrue((sig == 0).all())

    def test_all_buy_returns_buy(self):
        df = _make_df()
        rule = OrRule([_ConstRule(1, "a"), _ConstRule(1, "b")])
        sig = rule.generate_signal(df)
        self.assertTrue((sig == 1).all())

    def test_all_sell_returns_sell(self):
        df = _make_df()
        rule = OrRule([_ConstRule(-1, "a"), _ConstRule(-1, "b")])
        sig = rule.generate_signal(df)
        self.assertTrue((sig == -1).all())

    def test_high_threshold_requires_majority(self):
        """threshold=1.0 ならば全ルール一致でのみシグナル発生"""
        df = _make_df()
        rule = OrRule(
            [_ConstRule(1, "a"), _ConstRule(0, "b"), _ConstRule(0, "c")],
            threshold=1.0,
        )
        sig = rule.generate_signal(df)
        self.assertTrue((sig == 0).all())

    def test_threshold_exactly_met(self):
        """2/3 = 0.67 でシグナル発生 (threshold=0.5)"""
        df = _make_df()
        rule = OrRule(
            [_ConstRule(1, "a"), _ConstRule(1, "b"), _ConstRule(0, "c")],
            threshold=0.5,
        )
        sig = rule.generate_signal(df)
        self.assertTrue((sig == 1).all())

    def test_per_row_signals(self):
        df = _make_df(4)
        # 行ごとに異なる組み合わせ
        rule = OrRule(
            [
                _SeriesRule([1, 1, 0, -1], "r1"),
                _SeriesRule([0, 1, 0, -1], "r2"),
            ],
            threshold=0.5,
        )
        sig = rule.generate_signal(df)
        self.assertEqual(sig.iloc[0], 1)
        self.assertEqual(sig.iloc[1], 1)
        self.assertEqual(sig.iloc[2], 0)
        self.assertEqual(sig.iloc[3], -1)


class TestOrRuleNaming(unittest.TestCase):
    def test_auto_name(self):
        rule = OrRule([_ConstRule(1, "ema"), _ConstRule(1, "rsi")])
        self.assertIn("ema", rule.name)
        self.assertIn("rsi", rule.name)

    def test_custom_name(self):
        rule = OrRule([_ConstRule(1, "a")], name="my_or_rule")
        self.assertEqual(rule.name, "my_or_rule")

    def test_description_contains_threshold(self):
        rule = OrRule([_ConstRule(1, "a")], threshold=0.75)
        self.assertIn("75%", rule.description)


if __name__ == "__main__":
    unittest.main()
