import numpy as np
import pandas as pd


class SignalGenerator:
    def __init__(self, base_threshold: float = 0.005):
        """
        Args:
            base_threshold: 基準シグナル閾値（デフォルト ±0.5%）。
                            rolling_std が渡された場合はボラ比率で動的スケーリングされる。
        """
        self.base_threshold = base_threshold

    def generate_signal(
        self,
        data: pd.DataFrame,
        prediction: pd.Series,
        rolling_std: pd.Series = None,
    ) -> pd.Series:
        """
        テクニカル分析の結果とAI予測結果に基づいて売買シグナルを生成します。

        Args:
            data (pd.DataFrame): テクニカル分析の結果を含むデータフレーム。
                                 例: RSI, MACD, EMAなどの指標。
            prediction (pd.Series): AIモデルによる株価予測結果。
            rolling_std (pd.Series, optional): 予測値の直近ローリング標準偏差。
                渡された場合、閾値を `base_threshold * (rolling_std / avg_std)` で
                ボラティリティ連動補正する。None の場合は固定閾値を使用する。

        Returns:
            pd.Series: 各時点での売買シグナル ('Buy', 'Sell', 'Hold') 。
        """
        signals = pd.Series("Hold", index=data.index)

        # 動的閾値の計算
        if rolling_std is not None and len(rolling_std.dropna()) > 0:
            avg_std = rolling_std.mean()
            if avg_std > 0:
                vol_ratio = rolling_std / avg_std
                threshold = self.base_threshold * vol_ratio
            else:
                threshold = pd.Series(self.base_threshold, index=data.index)
        else:
            threshold = pd.Series(self.base_threshold, index=data.index)

        # 予測値が閾値を超えた場合に Buy/Sell シグナルを生成
        buy_condition = prediction > threshold
        sell_condition = prediction < -threshold

        signals.loc[buy_condition] = "Buy"
        signals.loc[sell_condition] = "Sell"

        # RSI判定（テスト用にカラム名を'RSI'に統一）
        if "RSI" in data.columns:
            # RSI極値でHoldゾーンを拡張:
            # 売られすぎ(RSI<30) かつ Hold なら Buy（底値拾い買いシグナル）
            # 買われすぎ(RSI>70) かつ Hold なら Sell（高値売りシグナル）
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
            "RSI": np.random.uniform(20, 80, 100),
        },
        index=dates,
    )

    # ダミーの予測結果 (株価変化率を想定)
    dummy_prediction = pd.Series(np.random.uniform(-0.01, 0.01, 100), index=dates)
    # ローリングSTD（直近20日）
    dummy_rolling_std = dummy_prediction.rolling(20).std()
    generator = SignalGenerator()
    signals_fixed = generator.generate_signal(dummy_data, dummy_prediction)
    signals_dynamic = generator.generate_signal(
        dummy_data, dummy_prediction, rolling_std=dummy_rolling_std
    )

    print("固定閾値シグナル:")
    print(signals_fixed.value_counts())
    print("\n動的閾値シグナル:")
    print(signals_dynamic.value_counts())
