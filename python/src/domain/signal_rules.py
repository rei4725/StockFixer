"""売買シグナルの判断の、domain の純粋なルール関数。

backtest と trading の両 BC から src.utils.signal_generator.SignalGenerator を通して使う。
しきい値と出来高フィルターの値はすべて引数で受け取り、設定やファイルは読まない。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def generate_trade_signals(
    data: pd.DataFrame,
    prediction: pd.Series,
    base_threshold: float,
    volume_window_days: int,
    volume_multiplier: float,
    rolling_std: Optional[pd.Series] = None,
) -> pd.Series:
    """
    テクニカル分析の結果とAI予測結果に基づいて売買シグナルを生成します。

    Args:
        data (pd.DataFrame): テクニカル分析の結果を含むデータフレーム。
                             例: RSI, MACD, EMAなどの指標。
        prediction (pd.Series): AIモデルによる株価予測結果。
        base_threshold: 基準シグナル閾値。
        volume_window_days: 出来高フィルターの平均を取る日数。
        volume_multiplier: 出来高フィルターの倍率。平均のこの倍未満なら Buy を Hold にする。
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
            threshold = base_threshold * vol_ratio
        else:
            threshold = pd.Series(base_threshold, index=data.index)
    else:
        threshold = pd.Series(base_threshold, index=data.index)

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

    # 出来高フィルター: 直近 volume_window_days 日平均の volume_multiplier 倍未満の場合はBuy→Hold
    if "Volume" in data.columns:
        avg_volume = data["Volume"].rolling(volume_window_days, min_periods=1).mean()
        low_volume = data["Volume"] < avg_volume * volume_multiplier
        signals.loc[low_volume & (signals == "Buy")] = "Hold"

    return signals
