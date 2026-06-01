"""ユニットテスト: src/backtest/rules/technical.py"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.backtest.rules.technical import (
    ALL_RULES,
    BollingerBandRule,
    EMAMomentumRule,
    MACDRSIRule,
    RSIContrarianRule,
    VolatilityBreakoutRule,
    VolumeBreakoutRule,
)


def _make_base(n: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.linspace(1000.0, 1100.0, n)
    return pd.DataFrame(
        {
            "Close": close,
            "Volume": np.full(n, 500_000.0),
        },
        index=idx,
    )


class TestVolumeBreakoutRule(unittest.TestCase):
    def test_returns_series_same_length(self):
        df = _make_base()
        rule = VolumeBreakoutRule()
        sig = rule.generate_signal(df)
        self.assertEqual(len(sig), len(df))

    def test_buy_on_volume_spike_and_price_breakout(self):
        n = 30
        df = _make_base(n)
        # 最終行: 出来高急増 + 価格が過去最高値更新
        df.loc[df.index[-1], "Volume"] = 5_000_000.0  # 通常の10倍
        df.loc[df.index[-1], "Close"] = 2000.0  # 過去最高値を大幅更新
        rule = VolumeBreakoutRule(volume_ratio=2.0, breakout_window=10)
        sig = rule.generate_signal(df)
        self.assertEqual(sig.iloc[-1], 1)

    def test_sell_on_price_breakdown(self):
        n = 30
        df = _make_base(n)
        # 最終行: 過去最安値を下回る
        df.loc[df.index[-1], "Close"] = 1.0
        rule = VolumeBreakoutRule(sell_window=5)
        sig = rule.generate_signal(df)
        self.assertEqual(sig.iloc[-1], -1)

    def test_no_signal_in_flat_market(self):
        n = 30
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame({"Close": np.full(n, 1000.0), "Volume": np.full(n, 500_000.0)}, index=idx)
        rule = VolumeBreakoutRule()
        sig = rule.generate_signal(df)
        # 出来高横ばい + 価格横ばい → buy シグナルなし
        self.assertTrue((sig <= 0).all())

    def test_signal_values_in_valid_range(self):
        sig = VolumeBreakoutRule().generate_signal(_make_base(50))
        self.assertTrue(sig.isin([-1, 0, 1]).all())


class TestEMAMomentumRule(unittest.TestCase):
    def test_returns_series_same_length(self):
        df = _make_base()
        sig = EMAMomentumRule().generate_signal(df)
        self.assertEqual(len(sig), len(df))

    def test_buy_when_ema_fast_above_slow_and_rising(self):
        n = 40
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Close": np.linspace(1000.0, 1200.0, n),  # 一貫上昇
                "ema_fast": np.linspace(1010.0, 1190.0, n),
                "ema_slow": np.linspace(1000.0, 1150.0, n),
            },
            index=idx,
        )
        rule = EMAMomentumRule()
        sig = rule.generate_signal(df)
        # 後半は上昇 + ema_fast > ema_slow → buy シグナルが存在
        self.assertIn(1, sig.values)

    def test_sell_when_ema_fast_below_slow(self):
        n = 40
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Close": np.linspace(1200.0, 1000.0, n),  # 下落
                "ema_fast": np.linspace(1100.0, 1000.0, n),
                "ema_slow": np.linspace(1200.0, 1100.0, n),
            },
            index=idx,
        )
        sig = EMAMomentumRule().generate_signal(df)
        self.assertIn(-1, sig.values)

    def test_fallback_ema_calculation_without_columns(self):
        """ema_fast / ema_slow 列がない場合は内部計算にフォールバック"""
        df = _make_base(50)
        sig = EMAMomentumRule(fast_window=5, slow_window=10).generate_signal(df)
        self.assertTrue(sig.isin([-1, 0, 1]).all())

    def test_signal_values_valid(self):
        sig = EMAMomentumRule().generate_signal(_make_base(50))
        self.assertTrue(sig.isin([-1, 0, 1]).all())


class TestRSIContrarianRule(unittest.TestCase):
    def test_no_rsi_column_returns_zero(self):
        df = _make_base()
        sig = RSIContrarianRule().generate_signal(df)
        self.assertTrue((sig == 0).all())

    def test_buy_on_oversold(self):
        n = 10
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame({"Close": np.full(n, 1000.0), "rsi": np.full(n, 20.0)}, index=idx)
        sig = RSIContrarianRule(oversold=30.0).generate_signal(df)
        self.assertTrue((sig == 1).all())

    def test_sell_on_overbought(self):
        n = 10
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame({"Close": np.full(n, 1000.0), "rsi": np.full(n, 80.0)}, index=idx)
        sig = RSIContrarianRule(overbought=70.0).generate_signal(df)
        self.assertTrue((sig == -1).all())

    def test_neutral_in_normal_range(self):
        n = 10
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame({"Close": np.full(n, 1000.0), "rsi": np.full(n, 50.0)}, index=idx)
        sig = RSIContrarianRule().generate_signal(df)
        self.assertTrue((sig == 0).all())

    def test_custom_thresholds(self):
        n = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame({"Close": [1000.0] * n, "rsi": [15.0, 25.0, 50.0, 75.0, 85.0]}, index=idx)
        rule = RSIContrarianRule(oversold=20.0, overbought=80.0)
        sig = rule.generate_signal(df)
        self.assertEqual(sig.iloc[0], 1)  # 15 < 20 → buy
        self.assertEqual(sig.iloc[1], 0)  # 25 → neutral
        self.assertEqual(sig.iloc[2], 0)  # 50 → neutral
        self.assertEqual(sig.iloc[3], 0)  # 75 → neutral
        self.assertEqual(sig.iloc[4], -1)  # 85 > 80 → sell


class TestBollingerBandRule(unittest.TestCase):
    def test_missing_columns_returns_zero(self):
        df = _make_base()
        sig = BollingerBandRule().generate_signal(df)
        self.assertTrue((sig == 0).all())

    def test_buy_below_lower_band(self):
        n = 10
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Close": np.full(n, 900.0),
                "bb_lower": np.full(n, 950.0),
                "bb_upper": np.full(n, 1100.0),
                "bb_middle": np.full(n, 1000.0),
            },
            index=idx,
        )
        sig = BollingerBandRule().generate_signal(df)
        self.assertTrue((sig == 1).all())

    def test_sell_above_middle_band_default(self):
        n = 10
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Close": np.full(n, 1050.0),
                "bb_lower": np.full(n, 900.0),
                "bb_upper": np.full(n, 1100.0),
                "bb_middle": np.full(n, 1000.0),
            },
            index=idx,
        )
        sig = BollingerBandRule(sell_at_upper=False).generate_signal(df)
        self.assertTrue((sig == -1).all())

    def test_sell_at_upper_band(self):
        n = 10
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Close": np.full(n, 1150.0),
                "bb_lower": np.full(n, 900.0),
                "bb_upper": np.full(n, 1100.0),
            },
            index=idx,
        )
        sig = BollingerBandRule(sell_at_upper=True).generate_signal(df)
        self.assertTrue((sig == -1).all())

    def test_bb_middle_computed_when_missing(self):
        """bb_middle 列がない場合は (bb_upper + bb_lower) / 2 で計算"""
        n = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Close": np.full(n, 1050.0),
                "bb_lower": np.full(n, 900.0),
                "bb_upper": np.full(n, 1100.0),
                # bb_middle なし → (900+1100)/2 = 1000
            },
            index=idx,
        )
        sig = BollingerBandRule(sell_at_upper=False).generate_signal(df)
        self.assertTrue((sig == -1).all())


class TestMACDRSIRule(unittest.TestCase):
    def test_missing_columns_returns_zero(self):
        df = _make_base()
        sig = MACDRSIRule().generate_signal(df)
        self.assertTrue((sig == 0).all())

    def test_buy_on_golden_cross_with_rsi_below_filter(self):
        n = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        # 1日目: macd <= signal, 2日目: macd > signal (ゴールデンクロス)
        df = pd.DataFrame(
            {
                "Close": [1000.0] * n,
                "macd": [-1.0, -0.5, 1.0, 0.5, 0.5],
                "macd_signal": [0.0, 0.0, 0.0, 0.0, 0.0],
                "rsi": [50.0] * n,  # 60 未満 → buy filter 通過
            },
            index=idx,
        )
        sig = MACDRSIRule(rsi_filter=60.0).generate_signal(df)
        self.assertEqual(sig.iloc[2], 1)  # ゴールデンクロス発生

    def test_sell_on_dead_cross(self):
        n = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Close": [1000.0] * n,
                "macd": [1.0, 0.5, -1.0, -0.5, -0.5],
                "macd_signal": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=idx,
        )
        sig = MACDRSIRule().generate_signal(df)
        self.assertEqual(sig.iloc[2], -1)  # デッドクロス

    def test_no_buy_when_rsi_above_filter(self):
        n = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Close": [1000.0] * n,
                "macd": [-1.0, -0.5, 1.0, 0.5, 0.5],
                "macd_signal": [0.0, 0.0, 0.0, 0.0, 0.0],
                "rsi": [70.0] * n,  # rsi_filter=60 を超える → buy 抑制
            },
            index=idx,
        )
        sig = MACDRSIRule(rsi_filter=60.0).generate_signal(df)
        self.assertNotEqual(sig.iloc[2], 1)

    def test_fallback_rsi_when_column_missing(self):
        """rsi 列がない場合は 50.0 でフォールバック（buy filter 通過）"""
        n = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Close": [1000.0] * n,
                "macd": [-1.0, -0.5, 1.0, 0.5, 0.5],
                "macd_signal": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=idx,
        )
        sig = MACDRSIRule(rsi_filter=60.0).generate_signal(df)
        self.assertEqual(sig.iloc[2], 1)


class TestVolatilityBreakoutRule(unittest.TestCase):
    def test_missing_atr_returns_zero(self):
        df = _make_base()
        sig = VolatilityBreakoutRule().generate_signal(df)
        self.assertTrue((sig == 0).all())

    def test_buy_on_upward_breakout(self):
        n = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Close": [1000.0, 1010.0, 1020.0, 1030.0, 1100.0],  # 最終行: 大幅上昇
                "atr": [20.0] * n,
            },
            index=idx,
        )
        rule = VolatilityBreakoutRule(buy_k=1.0)
        sig = rule.generate_signal(df)
        # prev_close=1030, prev_atr=20 → threshold=1050, close=1100 > 1050 → buy
        self.assertEqual(sig.iloc[-1], 1)

    def test_sell_on_downward_breakout(self):
        n = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Close": [1000.0, 990.0, 980.0, 970.0, 900.0],  # 最終行: 大幅下落
                "atr": [20.0] * n,
            },
            index=idx,
        )
        rule = VolatilityBreakoutRule(sell_k=1.5)
        sig = rule.generate_signal(df)
        # prev_close=970, prev_atr=20 → threshold=940, close=900 < 940 → sell
        self.assertEqual(sig.iloc[-1], -1)

    def test_neutral_in_normal_range(self):
        n = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {"Close": [1000.0, 1001.0, 1000.0, 1001.0, 1001.0], "atr": [20.0] * n},
            index=idx,
        )
        sig = VolatilityBreakoutRule(buy_k=1.0, sell_k=1.5).generate_signal(df)
        # 変動 1 pt < ATR 20 → buy/sell いずれも発生しない
        self.assertTrue((sig <= 0).all())


class TestAllRules(unittest.TestCase):
    def test_all_rules_instantiated(self):
        self.assertEqual(len(ALL_RULES), 6)

    def test_all_rules_have_name_and_description(self):
        for r in ALL_RULES:
            self.assertTrue(hasattr(r, "name"), f"{r} に name がない")
            self.assertTrue(hasattr(r, "description"), f"{r} に description がない")


if __name__ == "__main__":
    unittest.main()
