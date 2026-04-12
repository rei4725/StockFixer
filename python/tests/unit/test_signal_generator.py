from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import src.strategy.signal_generator as signal_generator_module
from src.strategy.signal_generator import (
    apply_multi_horizon_score_column,
    compute_multi_horizon_score,
)

# ---------------------------------------------------------------------------
# R-212: compute_multi_horizon_score
# ---------------------------------------------------------------------------


class TestComputeMultiHorizonScore(unittest.TestCase):
    def test_single_horizon_returns_diff_ratio(self):
        """3d/5d/10d がない場合は 1d の diff_ratio をそのまま返す"""
        score = compute_multi_horizon_score(diff_ratio_1d=0.05)
        self.assertAlmostEqual(score, 0.05)

    def test_multi_horizon_weighted_average(self):
        """全ホライズンが揃っている場合は重み付き平均になる（confluence なし）"""
        # 全ホライズンが同じ値 → 重み付き平均も同じ値
        score = compute_multi_horizon_score(0.04, 0.04, 0.04, 0.04)
        self.assertAlmostEqual(score, 0.04, places=6)

    def test_confluence_boost_amplifies_score(self):
        """confluence_score が大きいほどスコアが増幅される"""
        base = compute_multi_horizon_score(0.03, confluence_score=0)
        boosted = compute_multi_horizon_score(0.03, confluence_score=3)
        self.assertGreater(boosted, base)

    def test_negative_score_with_confluence(self):
        """売りシグナル（負値）でも confluence でさらに負方向に増幅される"""
        base = compute_multi_horizon_score(-0.03, confluence_score=0)
        boosted = compute_multi_horizon_score(-0.03, confluence_score=2)
        self.assertLess(boosted, base)

    def test_partial_horizons_weight_normalized(self):
        """一部ホライズンが None のとき残りの重みが正規化される"""
        # 1d(0.4)=0.02, 3d(0.3)=0.06 のみ有効 → 正規化後: 0.4/0.7*0.02 + 0.3/0.7*0.06
        score = compute_multi_horizon_score(0.02, diff_ratio_3d=0.06)
        expected = (0.4 / 0.7) * 0.02 + (0.3 / 0.7) * 0.06
        self.assertAlmostEqual(score, expected, places=6)

    def test_mixed_direction_reduces_score(self):
        """方向が混ざっている場合（confluence=0）はスコアが方向を薄める"""
        # 1d 上昇, 3d 下落 → 統合スコアは 1d のみより小さくなる
        score_1d_only = compute_multi_horizon_score(0.05)
        score_mixed = compute_multi_horizon_score(0.05, diff_ratio_3d=-0.05)
        self.assertGreater(score_1d_only, score_mixed)

    def test_returns_float(self):
        score = compute_multi_horizon_score(0.01, 0.02, 0.03, 0.04, confluence_score=2)
        self.assertIsInstance(score, float)


class TestApplyMultiHorizonScoreColumn(unittest.TestCase):
    def _make_df(self, **kwargs) -> pd.DataFrame:
        base = {
            "market": "jp",
            "symbol": "7203",
            "diff_ratio": 0.03,
        }
        base.update(kwargs)
        return pd.DataFrame([base])

    def test_adds_multi_horizon_score_column(self):
        df = self._make_df()
        result = apply_multi_horizon_score_column(df)
        self.assertIn("multi_horizon_score", result.columns)

    def test_score_equals_diff_ratio_when_no_horizons(self):
        """多ホライズン列がない場合は 1d の diff_ratio と等しい"""
        df = self._make_df()
        result = apply_multi_horizon_score_column(df)
        self.assertAlmostEqual(result["multi_horizon_score"].iloc[0], 0.03)

    def test_score_uses_all_horizons(self):
        """多ホライズン列が全て揃っている場合は統合スコアが計算される"""
        df = self._make_df(
            diff_ratio_3d=0.06,
            diff_ratio_5d=0.09,
            diff_ratio_10d=0.12,
            confluence_score=3,
        )
        result = apply_multi_horizon_score_column(df)
        # confluence ボーナスにより diff_ratio (0.03) より高くなるはず
        self.assertGreater(result["multi_horizon_score"].iloc[0], 0.03)

    def test_does_not_mutate_original(self):
        """元の DataFrame を変更しないこと"""
        df = self._make_df()
        original_cols = list(df.columns)
        apply_multi_horizon_score_column(df)
        self.assertEqual(list(df.columns), original_cols)

    def test_handles_none_and_nan_in_horizon_cols(self):
        """None や NaN の列があっても安全に動作すること"""
        import numpy as np

        df = self._make_df(diff_ratio_3d=float("nan"), diff_ratio_5d=None)
        result = apply_multi_horizon_score_column(df)
        score = result["multi_horizon_score"].iloc[0]
        self.assertFalse(np.isnan(score))

    def test_multiple_rows(self):
        rows = [
            {"market": "jp", "symbol": "7203", "diff_ratio": 0.05, "confluence_score": 2},
            {"market": "jp", "symbol": "6758", "diff_ratio": -0.03, "confluence_score": 1},
        ]
        df = pd.DataFrame(rows)
        result = apply_multi_horizon_score_column(df)
        self.assertEqual(len(result), 2)
        self.assertGreater(result["multi_horizon_score"].iloc[0], 0)
        self.assertLess(result["multi_horizon_score"].iloc[1], 0)


class TestSignalGenerator(unittest.TestCase):
    def setUp(self):
        """
        各テストメソッドの実行前に呼び出されます。
        テストに必要な共通のセットアップを行います。
        """
        self.signal_generator = signal_generator_module.SignalGenerator()

        # 再現性確保のためシードを固定した乱数生成器を使用
        rng = np.random.default_rng(seed=42)
        dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=10, freq="D"))
        self.dummy_data = pd.DataFrame(
            {
                "Open": 100 + rng.uniform(-0.5, 0.5, 10).cumsum(),
                "High": 101 + rng.uniform(-0.5, 0.5, 10).cumsum(),
                "Low": 99 + rng.uniform(-0.5, 0.5, 10).cumsum(),
                "Close": 100 + rng.uniform(-0.5, 0.5, 10).cumsum(),
                "Volume": rng.integers(1000, 5000, 10),
                "RSI": rng.uniform(20, 80, 10),
            },
            index=dates,
        )

        self.dummy_prediction = pd.Series(rng.uniform(-0.01, 0.01, 10), index=dates)

    def test_generate_signal_returns_series(self):
        """generate_signalメソッドがpandas.Seriesを返すことを確認します。"""
        signals = self.signal_generator.generate_signal(self.dummy_data, self.dummy_prediction)
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.dummy_data))

    def test_generate_signal_values(self):
        """生成されるシグナルが'Buy', 'Sell', 'Hold'のいずれかであることを確認します。"""
        signals = self.signal_generator.generate_signal(self.dummy_data, self.dummy_prediction)
        unique_signals = signals.unique()
        for signal in unique_signals:
            self.assertIn(signal, ["Buy", "Sell", "Hold"])

    def test_generate_signal_buy_condition(self):
        """
        Buyシグナルが正しく生成される条件をテストします。
        予測が0.005より大きい場合にBuyシグナルが生成されることを確認します。
        """
        # Buyシグナルが生成されるように予測値を設定
        buy_prediction = pd.Series([0.01] * len(self.dummy_data), index=self.dummy_data.index)
        signals = self.signal_generator.generate_signal(self.dummy_data, buy_prediction)
        self.assertTrue(all(signals == "Buy"))

    def test_generate_signal_sell_condition(self):
        """
        Sellシグナルが正しく生成される条件をテストします。
        予測が-0.005より小さい場合にSellシグナルが生成されることを確認します。
        """
        # Sellシグナルが生成されるように予測値を設定
        sell_prediction = pd.Series([-0.01] * len(self.dummy_data), index=self.dummy_data.index)
        signals = self.signal_generator.generate_signal(self.dummy_data, sell_prediction)
        self.assertTrue(all(signals == "Sell"))

    def test_generate_signal_hold_condition(self):
        """
        Holdシグナルが正しく生成される条件をテストします。
        中立な予測 + 中立なRSI(第30-70内)の場合にHoldシグナルが生成されることを確認。
        """
        data_neutral = self.dummy_data.copy()
        data_neutral["RSI"] = 50  # RSIを中立域に固定
        hold_prediction = pd.Series([0.001] * len(data_neutral), index=data_neutral.index)
        signals = self.signal_generator.generate_signal(data_neutral, hold_prediction)
        self.assertTrue(all(signals == "Hold"))

    def test_generate_signal_with_rsi(self):
        """RSIがシグナル生成に影響を与えることをテストします。"""
        # RSIが買われすぎで予測がBuyの場合
        data_high_rsi = self.dummy_data.copy()
        data_high_rsi["RSI"] = 75  # 買われすぎ
        prediction_buy = pd.Series([0.01] * len(data_high_rsi), index=data_high_rsi.index)
        signals_high_rsi = self.signal_generator.generate_signal(data_high_rsi, prediction_buy)
        self.assertTrue(all(signals_high_rsi == "Buy"))  # RSIが高くてもBuyは維持される

        # RSIが売られすぎで予測がSellの場合
        data_low_rsi = self.dummy_data.copy()
        data_low_rsi["RSI"] = 25  # 売られすぎ
        prediction_sell = pd.Series([-0.01] * len(data_low_rsi), index=data_low_rsi.index)
        signals_low_rsi = self.signal_generator.generate_signal(data_low_rsi, prediction_sell)
        self.assertTrue(all(signals_low_rsi == "Sell"))  # RSIが低くてもSellは維持される

        # RSIが中立域で予測が中立の場合
        data_neutral_rsi = self.dummy_data.copy()
        data_neutral_rsi["RSI"] = 50  # 中立
        prediction_neutral = pd.Series(
            [0.001] * len(data_neutral_rsi), index=data_neutral_rsi.index
        )
        signals_neutral_rsi = self.signal_generator.generate_signal(
            data_neutral_rsi, prediction_neutral
        )
        self.assertTrue(all(signals_neutral_rsi == "Hold"))

    def test_rsi_augments_hold_zone(self):
        """
        Holdゾーン(予測が中立)にRSI極値の補強シグナルをテスト:
        - RSI<30 + Hold予測 → Buy(売られすぎの押し目買いシグナル)
        - RSI>70 + Hold予測 → Sell(買われすぎの高値売りシグナル)
        """
        neutral_prediction = pd.Series([0.001] * len(self.dummy_data), index=self.dummy_data.index)

        # RSIが買われすぎ(>70) + 中立予測 → Sellに変換される
        data_overbought = self.dummy_data.copy()
        data_overbought["RSI"] = 80
        signals_ob = self.signal_generator.generate_signal(data_overbought, neutral_prediction)
        self.assertTrue(all(signals_ob == "Sell"))

        # RSIが売られすぎ(<30) + 中立予測 → Buyに変換される
        data_oversold = self.dummy_data.copy()
        data_oversold["RSI"] = 20
        signals_os = self.signal_generator.generate_signal(data_oversold, neutral_prediction)
        self.assertTrue(all(signals_os == "Buy"))

    def test_dynamic_threshold_with_rolling_std(self):
        """rolling_std を渡すと動的閾値（ボラティリティ連動）が計算されること"""
        dates = self.dummy_data.index
        # 強い買いシグナルを生成する予測値
        buy_prediction = pd.Series([0.05] * len(dates), index=dates)
        # avg_std > 0 になるよう差のある rolling_std を用意する
        rolling_std = pd.Series(
            [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010],
            index=dates,
        )

        signals = self.signal_generator.generate_signal(
            self.dummy_data, buy_prediction, rolling_std=rolling_std
        )
        # 動的閾値でも明確なBuy予測なのでBuyが含まれる
        self.assertIsInstance(signals, pd.Series)
        self.assertTrue((signals.isin(["Buy", "Sell", "Hold"])).all())

    def test_dynamic_threshold_with_zero_avg_std_fallback(self):
        """rolling_std の平均が0の場合は固定閾値にフォールバックすること"""
        dates = self.dummy_data.index
        # avg_std = 0 になるよう全て0の rolling_std
        rolling_std_zero = pd.Series([0.0] * len(dates), index=dates)
        buy_prediction = pd.Series([0.01] * len(dates), index=dates)

        signals = self.signal_generator.generate_signal(
            self.dummy_data, buy_prediction, rolling_std=rolling_std_zero
        )
        # 固定閾値フォールバックでも Buy が返ること
        self.assertTrue(all(signals == "Buy"))

    def test_dynamic_threshold_all_nan_rolling_std_fallback(self):
        """rolling_std が全て NaN の場合は固定閾値を使用すること"""
        dates = self.dummy_data.index
        rolling_std_nan = pd.Series([float("nan")] * len(dates), index=dates)
        sell_prediction = pd.Series([-0.01] * len(dates), index=dates)

        signals = self.signal_generator.generate_signal(
            self.dummy_data, sell_prediction, rolling_std=rolling_std_nan
        )
        self.assertTrue(all(signals == "Sell"))


if __name__ == "__main__":
    unittest.main()
