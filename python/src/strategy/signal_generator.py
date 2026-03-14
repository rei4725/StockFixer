import numpy as np
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
        signals = pd.Series("Hold", index=data.index)

        # 予測値が0.005 (0.5%)以上の上昇を予測した場合
        buy_condition = prediction > 0.005
        # 予測値が-0.005 (-0.5%)以下の下降を予測した場合
        sell_condition = prediction < -0.005

        signals.loc[buy_condition] = "Buy"
        signals.loc[sell_condition] = "Sell"

        # RSI判定（テスト仕様に合わせて列名を'RSI'に統一）
        if "RSI" in data.columns:
            # RSI極値でHoldゾーンを補強:
            # 売られすぎ(RSI<30) かつ Holdなら Buy（押し目買いシグナル）
            # 買われすぎ(RSI>70) かつ Holdなら Sell（高値売りシグナル）
            signals.loc[(data["RSI"] < 30) & (signals == "Hold")] = "Buy"
            signals.loc[(data["RSI"] > 70) & (signals == "Hold")] = "Sell"

        return signals


if __name__ == "__main__":
    # テスト用のダミーデータ
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=100, freq="D"))
    dummy_data = pd.DataFrame(
        {
            "Open": 100 + (np.random.rand(100) - 0.5).cumsum(),
            "High": 101 + (np.random.rand(100) - 0.5).cumsum(),
            "Low": 99 + (np.random.rand(100) - 0.5).cumsum(),
            "Close": 100 + (np.random.rand(100) - 0.5).cumsum(),
            "Volume": np.random.randint(1000, 5000, 100),
            "RSI": np.random.uniform(20, 80, 100),  # 仮のRSI値
        },
        index=dates,
    )

    # ダミーの予測結果 (価格変動率を想定)
    dummy_prediction = pd.Series(np.random.uniform(-0.01, 0.01, 100), index=dates)
    generator = SignalGenerator()
    signals = generator.generate_signal(dummy_data, dummy_prediction)

    print("生成されたシグナル:")
    print(signals.value_counts())
    print("\n最初の5つのシグナル:")
    print(signals.head())
    print("\n最後の5つのシグナル:")
    print(signals.tail())
