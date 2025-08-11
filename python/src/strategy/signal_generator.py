import pandas as pd

class SignalGenerator:
    def __init__(self):
        pass

    def generate_signal(self, data: pd.DataFrame, prediction: pd.Series) -> pd.Series:
        """
        テクニカル分析の結果とAI予測結果に基づいて売買シグナルを生成します。

        Args:
            data (pd.DataFrame): テクニカル分析の結果を含むデータフレーム。
                                 例: RSI, MACD, EMAなどの指標。
            prediction (pd.Series): AIモデルによる価格予測結果。

        Returns:
            pd.Series: 各時点での売買シグナル ('Buy', 'Sell', 'Hold')。
        """
        signals = pd.Series('Hold', index=data.index)

        # シグナル生成ロジック
        # 予測値とテクニカル指標を組み合わせて判断

        # 予測値が0.005 (0.5%)以上の上昇を予測した場合
        buy_condition = prediction > 0.005
        # 予測値が-0.005 (-0.5%)以下の下降を予測した場合
        sell_condition = prediction < -0.005

        signals.loc[buy_condition] = 'Buy'
        signals.loc[sell_condition] = 'Sell'

        # テクニカル指標を組み合わせる例 (RSIがデータに含まれていると仮定)
        if 'rsi' in data.columns:
            # RSIが30以下でBuyシグナルが出ている場合、Buyを強化
            signals.loc[(data['rsi'] <= 30) & (signals == 'Buy')] = 'Buy'
            # RSIが70以上でSellシグナルが出ている場合、Sellを強化
            signals.loc[(data['rsi'] >= 70) & (signals == 'Sell')] = 'Sell'

            # RSIが中立域 (30-70) で、かつBuy/Sellシグナルが出ていない場合はHold
            signals.loc[(data['rsi'] > 30) & (data['rsi'] < 70) & (signals == 'Hold')] = 'Hold'

        # MACDが利用可能な場合
        if 'macd' in data.columns and 'macd_signal' in data.columns:
            # MACDがシグナルラインを上抜いたら買い
            macd_buy_condition = (data['macd'] > data['macd_signal']) & (data['macd'].shift(1) <= data['macd_signal'].shift(1))
            signals.loc[macd_buy_condition] = 'Buy'

            # MACDがシグナルラインを下抜いたら売り
            macd_sell_condition = (data['macd'] < data['macd_signal']) & (data['macd'].shift(1) >= data['macd_signal'].shift(1))
            signals.loc[macd_sell_condition] = 'Sell'

        return signals

if __name__ == '__main__':
    # テスト用のダミーデータ
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D'))
    dummy_data = pd.DataFrame({
        'Open': 100 + (pd.np.random.rand(100) - 0.5).cumsum(),
        'High': 101 + (pd.np.random.rand(100) - 0.5).cumsum(),
        'Low': 99 + (pd.np.random.rand(100) - 0.5).cumsum(),
        'Close': 100 + (pd.np.random.rand(100) - 0.5).cumsum(),
        'Volume': pd.np.random.randint(1000, 5000, 100),
        'RSI': pd.np.random.uniform(20, 80, 100) # 仮のRSI値
    }, index=dates)

    # ダミーの予測結果 (価格変動率を想定)
    dummy_prediction = pd.Series(pd.np.random.uniform(-0.01, 0.01, 100), index=dates)

    generator = SignalGenerator()
    signals = generator.generate_signal(dummy_data, dummy_prediction)

    print("生成されたシグナル:")
    print(signals.value_counts())
    print("\n最初の5つのシグナル:")
    print(signals.head())
    print("\n最後の5つのシグナル:")
    print(signals.tail())
