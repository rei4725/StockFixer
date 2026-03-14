import unittest

import numpy as np
import pandas as pd

import src.strategy.signal_generator as signal_generator_module


class TestSignalGenerator(unittest.TestCase):
    def setUp(self):
        """
        各テストメソッドの実行前に呼び出されます。
        テストに必要な共通のセットアップを行います。
        """
        self.signal_generator = signal_generator_module.SignalGenerator()

        # テスト用のダミーデータフレームと予測シリーズを作成
        dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=10, freq="D"))
        self.dummy_data = pd.DataFrame(
            {
                "Open": 100 + (np.random.rand(10) - 0.5).cumsum(),
                "High": 101 + (np.random.rand(10) - 0.5).cumsum(),
                "Low": 99 + (np.random.rand(10) - 0.5).cumsum(),
                "Close": 100 + (np.random.rand(10) - 0.5).cumsum(),
                "Volume": np.random.randint(1000, 5000, 10),
                "RSI": np.random.uniform(20, 80, 10),
            },
            index=dates,
        )

        self.dummy_prediction = pd.Series(np.random.uniform(-0.01, 0.01, 10), index=dates)

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


if __name__ == "__main__":
    unittest.main()
