"""ユニットテスト: rule_engine/rules/composite.py + technical.py"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.rule_engine.rules.composite import AndRule, OrRule
from src.rule_engine.rules.technical import (
    ALL_RULES,
    BollingerBandRule,
    EMAMomentumRule,
    MACDRSIRule,
    RSIContrarianRule,
    VolatilityBreakoutRule,
    VolumeBreakoutRule,
)


def _make_df(n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"Close": np.linspace(100.0, 110.0, n)}, index=idx)


def _make_base(n: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.linspace(1000.0, 1100.0, n)
    return pd.DataFrame(
        {"Close": close, "Volume": np.full(n, 500_000.0)},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Stub rules for composite tests
# ---------------------------------------------------------------------------


class _ConstRule:
    def __init__(self, signal: int, name: str = "stub"):
        self._signal = signal
        self.name = name
        self.description = f"const_{signal}"

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self._signal, index=df.index)


class _SeriesRule:
    def __init__(self, signals: list[int], name: str = "series"):
        self._signals = signals
        self.name = name
        self.description = "series"

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self._signals, index=df.index)


# ---------------------------------------------------------------------------
# AndRule
# ---------------------------------------------------------------------------


class TestAndRule(unittest.TestCase):
    def test_all_buy_returns_buy(self):
        df = _make_df()
        rule = AndRule([_ConstRule(1, "a"), _ConstRule(1, "b")])
        self.assertTrue((rule.generate_signal(df) == 1).all())

    def test_partial_buy_returns_neutral(self):
        df = _make_df()
        rule = AndRule([_ConstRule(1, "a"), _ConstRule(0, "b")])
        self.assertTrue((rule.generate_signal(df) == 0).all())

    def test_any_sell_gives_sell(self):
        df = _make_df()
        rule = AndRule([_ConstRule(1, "a"), _ConstRule(-1, "b")])
        self.assertTrue((rule.generate_signal(df) == -1).all())

    def test_all_sell_returns_sell(self):
        df = _make_df()
        rule = AndRule([_ConstRule(-1, "a"), _ConstRule(-1, "b")])
        self.assertTrue((rule.generate_signal(df) == -1).all())

    def test_auto_name_joined_from_children(self):
        rule = AndRule([_ConstRule(1, "ema"), _ConstRule(1, "rsi")])
        self.assertIn("ema", rule.name)
        self.assertIn("rsi", rule.name)

    def test_custom_name(self):
        rule = AndRule([_ConstRule(1, "x")], name="custom")
        self.assertEqual(rule.name, "custom")

    def test_description_contains_child(self):
        rule = AndRule([_ConstRule(1, "a"), _ConstRule(1, "b")])
        self.assertIn("const_1", rule.description)


# ---------------------------------------------------------------------------
# OrRule
# ---------------------------------------------------------------------------


class TestOrRule(unittest.TestCase):
    def test_one_of_two_buy_threshold_half(self):
        df = _make_df()
        rule = OrRule([_ConstRule(1, "a"), _ConstRule(0, "b")], threshold=0.5)
        self.assertTrue((rule.generate_signal(df) == 1).all())

    def test_all_neutral_returns_neutral(self):
        df = _make_df()
        rule = OrRule([_ConstRule(0, "a"), _ConstRule(0, "b")])
        self.assertTrue((rule.generate_signal(df) == 0).all())

    def test_threshold_1_requires_all_agree(self):
        df = _make_df()
        rule = OrRule(
            [_ConstRule(1, "a"), _ConstRule(0, "b"), _ConstRule(0, "c")],
            threshold=1.0,
        )
        self.assertTrue((rule.generate_signal(df) == 0).all())

    def test_majority_buy_gives_buy(self):
        df = _make_df()
        rule = OrRule(
            [_ConstRule(1, "a"), _ConstRule(1, "b"), _ConstRule(0, "c")],
            threshold=0.5,
        )
        self.assertTrue((rule.generate_signal(df) == 1).all())

    def test_per_row_signals(self):
        df = _make_df(3)
        rule = OrRule(
            [_SeriesRule([1, 0, -1], "r1"), _SeriesRule([1, 0, -1], "r2")],
            threshold=0.5,
        )
        sig = rule.generate_signal(df)
        self.assertEqual(sig.iloc[0], 1)
        self.assertEqual(sig.iloc[1], 0)
        self.assertEqual(sig.iloc[2], -1)

    def test_auto_name_joined_from_children(self):
        rule = OrRule([_ConstRule(1, "vol"), _ConstRule(1, "rsi")])
        self.assertIn("vol", rule.name)

    def test_custom_name(self):
        rule = OrRule([_ConstRule(1, "a")], name="my_or")
        self.assertEqual(rule.name, "my_or")

    def test_description_contains_threshold_percent(self):
        rule = OrRule([_ConstRule(1, "a")], threshold=0.6)
        self.assertIn("60%", rule.description)


# ---------------------------------------------------------------------------
# VolumeBreakoutRule
# ---------------------------------------------------------------------------


class TestVolumeBreakoutRule(unittest.TestCase):
    def test_signal_in_valid_range(self):
        sig = VolumeBreakoutRule().generate_signal(_make_base(50))
        self.assertTrue(sig.isin([-1, 0, 1]).all())

    def test_buy_on_volume_spike_and_new_high(self):
        df = _make_base(30)
        df.loc[df.index[-1], "Volume"] = 5_000_000.0
        df.loc[df.index[-1], "Close"] = 9999.0
        sig = VolumeBreakoutRule(volume_ratio=2.0, breakout_window=10).generate_signal(df)
        self.assertEqual(sig.iloc[-1], 1)

    def test_sell_on_new_low(self):
        df = _make_base(20)
        df.loc[df.index[-1], "Close"] = 1.0
        sig = VolumeBreakoutRule(sell_window=5).generate_signal(df)
        self.assertEqual(sig.iloc[-1], -1)

    def test_flat_market_no_buy(self):
        n = 30
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame({"Close": np.full(n, 1000.0), "Volume": np.full(n, 500_000.0)}, index=idx)
        sig = VolumeBreakoutRule().generate_signal(df)
        self.assertTrue((sig <= 0).all())


# ---------------------------------------------------------------------------
# EMAMomentumRule
# ---------------------------------------------------------------------------


class TestEMAMomentumRule(unittest.TestCase):
    def test_buy_when_fast_above_slow_and_close_above_fast_rising(self):
        n = 40
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Close": np.linspace(1000.0, 1200.0, n),
                "ema_fast": np.linspace(1010.0, 1190.0, n),
                "ema_slow": np.linspace(1000.0, 1150.0, n),
            },
            index=idx,
        )
        sig = EMAMomentumRule().generate_signal(df)
        self.assertIn(1, sig.values)

    def test_sell_when_fast_below_slow(self):
        n = 40
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Close": np.linspace(1200.0, 1000.0, n),
                "ema_fast": np.linspace(1100.0, 950.0, n),
                "ema_slow": np.linspace(1200.0, 1100.0, n),
            },
            index=idx,
        )
        sig = EMAMomentumRule().generate_signal(df)
        self.assertIn(-1, sig.values)

    def test_fallback_ema_without_columns(self):
        sig = EMAMomentumRule(fast_window=5, slow_window=10).generate_signal(_make_base(50))
        self.assertTrue(sig.isin([-1, 0, 1]).all())


# ---------------------------------------------------------------------------
# RSIContrarianRule
# ---------------------------------------------------------------------------


class TestRSIContrarianRule(unittest.TestCase):
    def test_no_rsi_column_returns_zero(self):
        sig = RSIContrarianRule().generate_signal(_make_base())
        self.assertTrue((sig == 0).all())

    def test_buy_oversold(self):
        n, idx = 5, pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame({"Close": [1000.0] * n, "rsi": [20.0] * n}, index=idx)
        self.assertTrue((RSIContrarianRule(oversold=30.0).generate_signal(df) == 1).all())

    def test_sell_overbought(self):
        n, idx = 5, pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame({"Close": [1000.0] * n, "rsi": [80.0] * n}, index=idx)
        self.assertTrue((RSIContrarianRule(overbought=70.0).generate_signal(df) == -1).all())

    def test_neutral_mid_range(self):
        n, idx = 5, pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame({"Close": [1000.0] * n, "rsi": [50.0] * n}, index=idx)
        self.assertTrue((RSIContrarianRule().generate_signal(df) == 0).all())


# ---------------------------------------------------------------------------
# BollingerBandRule
# ---------------------------------------------------------------------------


class TestBollingerBandRule(unittest.TestCase):
    def test_missing_columns_returns_zero(self):
        sig = BollingerBandRule().generate_signal(_make_base())
        self.assertTrue((sig == 0).all())

    def test_buy_below_lower_band(self):
        n, idx = 5, pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {"Close": [900.0] * n, "bb_lower": [950.0] * n, "bb_upper": [1100.0] * n},
            index=idx,
        )
        self.assertTrue((BollingerBandRule().generate_signal(df) == 1).all())

    def test_sell_above_middle_default(self):
        n, idx = 5, pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "Close": [1050.0] * n,
                "bb_lower": [900.0] * n,
                "bb_upper": [1100.0] * n,
                "bb_middle": [1000.0] * n,
            },
            index=idx,
        )
        self.assertTrue((BollingerBandRule(sell_at_upper=False).generate_signal(df) == -1).all())

    def test_sell_at_upper_flag(self):
        n, idx = 5, pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {"Close": [1150.0] * n, "bb_lower": [900.0] * n, "bb_upper": [1100.0] * n},
            index=idx,
        )
        self.assertTrue((BollingerBandRule(sell_at_upper=True).generate_signal(df) == -1).all())

    def test_bb_middle_computed_from_bands(self):
        n, idx = 5, pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {"Close": [1050.0] * n, "bb_lower": [900.0] * n, "bb_upper": [1100.0] * n},
            index=idx,
        )
        # bb_middle computed = 1000, close 1050 > 1000 → sell
        self.assertTrue((BollingerBandRule(sell_at_upper=False).generate_signal(df) == -1).all())


# ---------------------------------------------------------------------------
# MACDRSIRule
# ---------------------------------------------------------------------------


class TestMACDRSIRule(unittest.TestCase):
    def test_missing_columns_returns_zero(self):
        sig = MACDRSIRule().generate_signal(_make_base())
        self.assertTrue((sig == 0).all())

    def test_buy_on_golden_cross(self):
        n = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Close": [1000.0] * n,
                "macd": [-1.0, -0.5, 1.0, 0.5, 0.5],
                "macd_signal": [0.0] * n,
                "rsi": [50.0] * n,
            },
            index=idx,
        )
        sig = MACDRSIRule(rsi_filter=60.0).generate_signal(df)
        self.assertEqual(sig.iloc[2], 1)

    def test_sell_on_dead_cross(self):
        n = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Close": [1000.0] * n,
                "macd": [1.0, 0.5, -1.0, -0.5, -0.5],
                "macd_signal": [0.0] * n,
            },
            index=idx,
        )
        self.assertEqual(MACDRSIRule().generate_signal(df).iloc[2], -1)

    def test_no_buy_when_rsi_too_high(self):
        n = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Close": [1000.0] * n,
                "macd": [-1.0, -0.5, 1.0, 0.5, 0.5],
                "macd_signal": [0.0] * n,
                "rsi": [70.0] * n,
            },
            index=idx,
        )
        self.assertNotEqual(MACDRSIRule(rsi_filter=60.0).generate_signal(df).iloc[2], 1)

    def test_fallback_rsi_50_when_column_missing(self):
        n = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Close": [1000.0] * n,
                "macd": [-1.0, -0.5, 1.0, 0.5, 0.5],
                "macd_signal": [0.0] * n,
            },
            index=idx,
        )
        self.assertEqual(MACDRSIRule(rsi_filter=60.0).generate_signal(df).iloc[2], 1)


# ---------------------------------------------------------------------------
# VolatilityBreakoutRule
# ---------------------------------------------------------------------------


class TestVolatilityBreakoutRule(unittest.TestCase):
    def test_missing_atr_returns_zero(self):
        sig = VolatilityBreakoutRule().generate_signal(_make_base())
        self.assertTrue((sig == 0).all())

    def test_buy_on_upward_breakout(self):
        n = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {"Close": [1000.0, 1010.0, 1020.0, 1030.0, 1100.0], "atr": [20.0] * n},
            index=idx,
        )
        self.assertEqual(VolatilityBreakoutRule(buy_k=1.0).generate_signal(df).iloc[-1], 1)

    def test_sell_on_downward_breakout(self):
        n = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {"Close": [1000.0, 990.0, 980.0, 970.0, 900.0], "atr": [20.0] * n},
            index=idx,
        )
        self.assertEqual(VolatilityBreakoutRule(sell_k=1.5).generate_signal(df).iloc[-1], -1)

    def test_neutral_in_range(self):
        n = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {"Close": [1000.0, 1001.0, 1000.0, 1001.0, 1001.0], "atr": [20.0] * n},
            index=idx,
        )
        sig = VolatilityBreakoutRule(buy_k=1.0, sell_k=1.5).generate_signal(df)
        self.assertTrue((sig == 0).all())


# ---------------------------------------------------------------------------
# ALL_RULES
# ---------------------------------------------------------------------------


class TestAllRules(unittest.TestCase):
    def test_count(self):
        self.assertEqual(len(ALL_RULES), 6)

    def test_each_has_name_and_description(self):
        for r in ALL_RULES:
            self.assertTrue(hasattr(r, "name"))
            self.assertTrue(hasattr(r, "description"))


if __name__ == "__main__":
    unittest.main()
