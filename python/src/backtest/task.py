"""
バックテストタスク定義

推論対象（ラベル定義 + シグナル変換）をプロトコルとして抽象化する。
新しい推論タスクはこのプロトコルを実装すれば Backtester に差し込める。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


def _build_threshold_series(
    pred: pd.Series,
    base_threshold: float,
    window: int = 20,
    min_scale: float = 0.5,
    max_scale: float = 2.0,
) -> pd.Series:
    """予測値の分散に応じて売買閾値を動的調整する。"""
    threshold = pd.Series(float(base_threshold), index=pred.index, dtype=float)
    if base_threshold <= 0 or len(pred) < window:
        return threshold

    min_periods = min(window, max(5, window // 2))
    rolling_std = pred.rolling(window=window, min_periods=min_periods).std()
    valid_std = rolling_std.dropna()
    if valid_std.empty:
        return threshold

    positive_std = valid_std[valid_std > 0]
    if positive_std.empty:
        return threshold

    baseline = float(positive_std.median())
    if baseline <= 0:
        return threshold

    scale = (rolling_std / baseline).clip(lower=min_scale, upper=max_scale).fillna(1.0)
    return threshold * scale


@runtime_checkable
class BacktestTask(Protocol):
    """
    バックテストタスクのインターフェース。

    Backtester はこのプロトコルに準拠したオブジェクトを受け取り、
    ラベル生成・シグナル変換を委譲する。
    """

    label_col: str
    """DataFrameに追加するラベル列名"""

    def make_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        価格 DataFrame からラベル（教師信号）を生成する。

        Args:
            df: OHLCV + テクニカル指標 DataFrame

        Returns:
            ラベルの pd.Series（インデックスは df と揃える）
        """
        ...

    def make_signal(self, pred: pd.Series) -> pd.Series:
        """
        モデルの予測値からトレードシグナルを生成する。

        Args:
            pred: モデルが出力した予測値 Series

        Returns:
            シグナル Series: 1=buy, -1=sell, 0=hold
        """
        ...


class ReturnRegressionTask:
    """
    翌日リターン回帰タスク。

    ラベル: 翌日変化率 (Close_t+1 - Close_t) / Close_t
    シグナル: pred > threshold → buy(1), pred < -threshold → sell(-1), else hold(0)

    これは既存パイプラインの `y` と同一の定義。
    """

    label_col: str = "y"

    def __init__(
        self,
        threshold: float = 0.0,
        adaptive_window: int = 20,
        adaptive_min_scale: float = 0.5,
        adaptive_max_scale: float = 2.0,
    ):
        """
        Args:
            threshold: シグナル発生の閾値（予測変化率の絶対値）。
                       0.0 の場合は pred > 0 で buy、pred < 0 で sell。
            adaptive_window: 動的閾値算出に使う予測分散の窓長。
            adaptive_min_scale: 閾値スケールの下限。
            adaptive_max_scale: 閾値スケールの上限。
        """
        self.threshold = threshold
        self.adaptive_window = adaptive_window
        self.adaptive_min_scale = adaptive_min_scale
        self.adaptive_max_scale = adaptive_max_scale

    def make_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Close 列の翌日変化率を返す。

        Args:
            df: Close 列を含む DataFrame（インデックスは日付）

        Returns:
            翌日変化率の Series（最終行は NaN）
        """
        close = df["Close"] if "Close" in df.columns else df["close"]
        return close.pct_change().shift(-1).rename(self.label_col)

    def make_signal(self, pred: pd.Series) -> pd.Series:
        """
        予測変化率から売買シグナルを生成する。

        Args:
            pred: 予測変化率の Series

        Returns:
            シグナル Series (1=buy, -1=sell, 0=hold)
        """
        signal = pd.Series(0, index=pred.index, name="signal")
        threshold = _build_threshold_series(
            pred,
            self.threshold,
            window=self.adaptive_window,
            min_scale=self.adaptive_min_scale,
            max_scale=self.adaptive_max_scale,
        )
        signal[pred > threshold] = 1
        signal[pred < -threshold] = -1
        return signal
