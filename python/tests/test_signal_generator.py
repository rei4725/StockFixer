import unittest
import pandas as pd
import numpy as np
from python.src.strategy.signal_generator.signal_generator import SignalGenerator

class TestSignalGenerator(unittest.TestCase):

    def setUp(self):
        """
        各テストメソッドの実行前に呼び出されます。
        テストに必要な共通のセットアップを行います。
        """
        self.signal_generator = SignalGenerator()
        
        # テスト用のダミーデータフレームと予測シリーズを作成
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=10, freq='D'))
        self.dummy_data = pd.DataFrame({
            'Open': 100 + (np.random.rand(10) - 0.5).cumsum(),
            'High': 101 + (np.random.rand(10) - 0.5).cumsum(),
            'Low': 99 + (np.random.rand(10) - 0.5).cumsum(),
            'Close': 100 + (np.random.rand(10) - 0.5).cumsum(),
            'Volume': np.random.randint(1000, 5000, 10),
            'RSI': np.random.uniform(20, 80, 10)
        }, index=dates)
        
        self.dummy_prediction = pd.Series(np.random.uniform(-0.01, 0.01, 10), index=dates)

    def test_generate_signal_returns_series(self):
        """
        generate_signalメソッドがpandas.Seriesを返すことを確認します。
        """
        signals = self.signal_generator.generate_signal(self.dummy_data, self.dummy_prediction)
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.dummy_data))

    def test_generate_signal_values(self):
        """
        生成されるシグナルが'Buy', 'Sell', 'Hold'のいずれかであることを確認します。
        """
        signals = self.signal_generator.generate_signal(self.dummy_data, self.dummy_prediction)
        unique_signals = signals.unique()
        for signal in unique_signals:
            self.assertIn(signal, ['Buy', 'Sell', 'Hold'])

    def test_generate_signal_buy_condition(self):
        """
        Buyシグナルが正しく生成される条件をテストします。
        予測が0.005より大きい場合にBuyシグナルが生成されることを確認します。
        """
        # Buyシグナルが生成されるように予測値を設定
        buy_prediction = pd.Series([0.01] * len(self.dummy_data), index=self.dummy_data.index)
        signals = self.signal_generator.generate_signal(self.dummy_data, buy_prediction)
        self.assertTrue(all(signals == 'Buy'))

    def test_generate_signal_sell_condition(self):
        """
        Sellシグナルが正しく生成される条件をテストします。
        予測が-0.005より小さい場合にSellシグナルが生成されることを確認します。
        """
        # Sellシグナルが生成されるように予測値を設定
        sell_prediction = pd.Series([-0.01] * len(self.dummy_data), index=self.dummy_data.index)
        signals = self.signal_generator.generate_signal(self.dummy_data, sell_prediction)
        self.assertTrue(all(signals == 'Sell'))

    def test_generate_signal_hold_condition(self):
        """
        Holdシグナルが正しく生成される条件をテストします。
        予測が中立的な場合にHoldシグナルが生成されることを確認します。
        """
        # Holdシグナルが生成されるように予測値を設定
        hold_prediction = pd.Series([0.001] * len(self.dummy_data), index=self.dummy_data.index)
        signals = self.signal_generator.generate_signal(self.dummy_data, hold_prediction)
        # RSIが中立域の場合、Holdになることを確認
        self.assertTrue(all(signals == 'Hold'))

    def test_generate_signal_with_rsi(self):
        """
        RSIがシグナル生成に影響を与えることをテストします。
        """
        # RSIが買われすぎで予測がBuyの場合
        data_high_rsi = self.dummy_data.copy()
        data_high_rsi['RSI'] = 75 # 買われすぎ
        prediction_buy = pd.Series([0.01] * len(data_high_rsi), index=data_high_rsi.index)
        signals_high_rsi = self.signal_generator.generate_signal(data_high_rsi, prediction_buy)
        self.assertTrue(all(signals_high_rsi == 'Buy')) # RSIが高くてもBuyは維持される

        # RSIが売られすぎで予測がSellの場合
        data_low_rsi = self.dummy_data.copy()
        data_low_rsi['RSI'] = 25 # 売られすぎ
        prediction_sell = pd.Series([-0.01] * len(data_low_rsi), index=data_low_rsi.index)
        signals_low_rsi = self.signal_generator.generate_signal(data_low_rsi, prediction_sell)
        self.assertTrue(all(signals_low_rsi == 'Sell')) # RSIが低くてもSellは維持される

        # RSIが中立域で予測が中立の場合
        data_neutral_rsi = self.dummy_data.copy()
        data_neutral_rsi['RSI'] = 50 # 中立
        prediction_neutral = pd.Series([0.001] * len(data_neutral_rsi), index=data_neutral_rsi.index)
        signals_neutral_rsi = self.signal_generator.generate_signal(data_neutral_rsi, prediction_neutral)
        self.assertTrue(all(signals_neutral_rsi == 'Hold'))

if __name__ == '__main__':
    unittest.main()
