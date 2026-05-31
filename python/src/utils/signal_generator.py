"""売買シグナル生成ユーティリティ

trading BC と backtest BC の両方から参照できる SignalGenerator 本体。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from config.settings import VOLUME_FILTER_MULTIPLIER, VOLUME_FILTER_WINDOW_DAYS
from src.utils.logger import get_logger
from src.utils.optimal_params_loader import get_optimal_params

logger = get_logger(__name__)


class SignalGenerator:
    def __init__(
        self,
        base_threshold: float = 0.005,
        market: Optional[str] = None,
        symbol: Optional[str] = None,
    ):
        """
        Args:
            base_threshold: 基準シグナル閾値（デフォルト ±0.5%）。
                            rolling_std が渡された場合はボラ比率で動的スケーリングされる。
            market: マーケット識別子（例: "jp", "us"）。
                    symbol と合わせて指定すると optimal_params.json から threshold を自動ロードする。
            symbol: 銘柄シンボル（例: "7203", "AAPL"）。
        """
        if market is not None and symbol is not None:
            params = get_optimal_params(market, symbol)
            if params is not None and "threshold" in params:
                self.base_threshold = float(params["threshold"])
                logger.debug(
                    f"[{market}_{symbol}] optimal_params.json から threshold を自動ロード: "
                    f"{self.base_threshold}"
                )
            else:
                logger.warning(
                    f"[{market}_{symbol}] optimal_params.json が見つからないか threshold が未設定のため "
                    f"デフォルト値 {base_threshold} を使用します"
                )
                self.base_threshold = base_threshold
        else:
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

        # 出来高フィルター: 直近20日平均の1.5倍未満の場合はBuy→Hold
        if "Volume" in data.columns:
            avg_volume = data["Volume"].rolling(VOLUME_FILTER_WINDOW_DAYS, min_periods=1).mean()
            low_volume = data["Volume"] < avg_volume * VOLUME_FILTER_MULTIPLIER
            signals.loc[low_volume & (signals == "Buy")] = "Hold"

        return signals
