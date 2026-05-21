"""
テクニカル指標ベースの売買ルール実装

add_technical_indicators() で付与された列（rsi, macd, ema_fast, ema_slow,
bb_upper, bb_lower, atr 等）を使ってシグナルを生成する。
"""

from __future__ import annotations

import pandas as pd


class VolumeBreakoutRule:
    """
    出来高ブレイクアウト

    - Buy: 出来高が直近MA比で急増 AND 終値が直近N日高値を更新
    - Sell: 終値が直近M日安値を下回る（トレンド崩壊）
    """

    name = "volume_breakout"
    description = "出来高急増 + 直近高値ブレイクアウト"

    def __init__(
        self,
        volume_ratio: float = 2.0,
        volume_ma_window: int = 20,
        breakout_window: int = 20,
        sell_window: int = 10,
    ):
        self.volume_ratio = volume_ratio
        self.volume_ma_window = volume_ma_window
        self.breakout_window = breakout_window
        self.sell_window = sell_window

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]
        volume = df["Volume"]

        vol_ma = volume.rolling(self.volume_ma_window, min_periods=5).mean()
        vol_spike = volume > vol_ma * self.volume_ratio

        rolling_high = close.rolling(self.breakout_window, min_periods=5).max().shift(1)
        price_breakout = close >= rolling_high

        rolling_low = close.rolling(self.sell_window, min_periods=3).min().shift(1)
        price_breakdown = close < rolling_low

        signal = pd.Series(0, index=df.index)
        signal[vol_spike & price_breakout] = 1
        signal[price_breakdown] = -1
        return signal


class EMAMomentumRule:
    """
    EMAモメンタム

    - Buy: EMA短期 > EMA長期 AND 終値 > EMA短期 AND 前日比プラス
    - Sell: EMA短期 < EMA長期（トレンド反転）
    """

    name = "ema_momentum"
    description = "EMAゴールデンクロス + 終値がEMA上方"

    def __init__(self, fast_window: int = 12, slow_window: int = 26):
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]

        if "ema_fast" in df.columns and "ema_slow" in df.columns:
            ema_fast = df["ema_fast"]
            ema_slow = df["ema_slow"]
        else:
            ema_fast = close.ewm(span=self.fast_window, adjust=False).mean()
            ema_slow = close.ewm(span=self.slow_window, adjust=False).mean()

        daily_return = close.pct_change()
        bullish = (ema_fast > ema_slow) & (close > ema_fast) & (daily_return > 0)
        bearish = ema_fast < ema_slow

        signal = pd.Series(0, index=df.index)
        signal[bullish] = 1
        signal[bearish] = -1
        return signal


class RSIContrarianRule:
    """
    RSI逆張り

    - Buy: RSI が過売水準を下回る（売られすぎ）
    - Sell: RSI が過買水準を上回る（買われすぎ）
    """

    name = "rsi_contrarian"
    description = "RSI逆張り（売られすぎBuy / 買われすぎSell）"

    def __init__(self, oversold: float = 30.0, overbought: float = 70.0):
        self.oversold = oversold
        self.overbought = overbought

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        if "rsi" not in df.columns:
            return pd.Series(0, index=df.index)

        rsi = df["rsi"]
        signal = pd.Series(0, index=df.index)
        signal[rsi < self.oversold] = 1
        signal[rsi > self.overbought] = -1
        return signal


class BollingerBandRule:
    """
    ボリンジャーバンド逆張り

    - Buy: 終値がBB下限を下回る（過売）
    - Sell: 終値がBB中線を超える（平均回帰完了）
    """

    name = "bollinger_band"
    description = "ボリンジャーバンド逆張り（下限Buy / 中線越えSell）"

    def __init__(self, sell_at_upper: bool = False):
        self.sell_at_upper = sell_at_upper

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        if "bb_lower" not in df.columns or "bb_upper" not in df.columns:
            return pd.Series(0, index=df.index)

        close = df["Close"]
        bb_lower = df["bb_lower"]
        bb_upper = df["bb_upper"]
        bb_middle = df.get("bb_middle", (bb_upper + bb_lower) / 2)

        buy_cond = close < bb_lower
        sell_cond = close > bb_upper if self.sell_at_upper else close > bb_middle

        signal = pd.Series(0, index=df.index)
        signal[buy_cond] = 1
        signal[sell_cond] = -1
        return signal


class MACDRSIRule:
    """
    MACD + RSIフィルター複合ルール

    - Buy: MACDラインが信号線をゴールデンクロス AND RSI < rsi_filter
    - Sell: MACDラインが信号線をデッドクロス
    """

    name = "macd_rsi"
    description = "MACDゴールデンクロス + RSIフィルター"

    def __init__(self, rsi_filter: float = 60.0):
        self.rsi_filter = rsi_filter

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        if "macd" not in df.columns or "macd_signal" not in df.columns:
            return pd.Series(0, index=df.index)

        macd = df["macd"]
        macd_sig = df["macd_signal"]
        rsi = df.get("rsi", pd.Series(50.0, index=df.index))

        # クロス検出: 前日はmacd<=signal、今日はmacd>signal
        macd_prev = macd.shift(1)
        sig_prev = macd_sig.shift(1)

        golden_cross = (macd_prev <= sig_prev) & (macd > macd_sig)
        dead_cross = (macd_prev >= sig_prev) & (macd < macd_sig)

        buy_cond = golden_cross & (rsi < self.rsi_filter)
        sell_cond = dead_cross

        signal = pd.Series(0, index=df.index)
        signal[buy_cond] = 1
        signal[sell_cond] = -1
        return signal


class VolatilityBreakoutRule:
    """
    ATRボラティリティブレイクアウト

    - Buy: 終値が前日終値 + k * ATR を超える（上方ブレイク）
    - Sell: 終値が前日終値 - stop_k * ATR を下回る（下方ブレイク、ロスカット）
    """

    name = "volatility_breakout"
    description = "ATRボラティリティブレイクアウト（上方突破Buy）"

    def __init__(self, buy_k: float = 1.0, sell_k: float = 1.5):
        self.buy_k = buy_k
        self.sell_k = sell_k

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        if "atr" not in df.columns:
            return pd.Series(0, index=df.index)

        close = df["Close"]
        atr = df["atr"]
        prev_close = close.shift(1)
        prev_atr = atr.shift(1)

        buy_cond = close > prev_close + self.buy_k * prev_atr
        sell_cond = close < prev_close - self.sell_k * prev_atr

        signal = pd.Series(0, index=df.index)
        signal[buy_cond] = 1
        signal[sell_cond] = -1
        return signal


ALL_RULES: list = [
    VolumeBreakoutRule(),
    EMAMomentumRule(),
    RSIContrarianRule(),
    BollingerBandRule(),
    MACDRSIRule(),
    VolatilityBreakoutRule(),
]
