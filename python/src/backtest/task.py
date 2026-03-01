"""
バックテストタスク定義

推論対象（ラベル定義 + シグナル変換）をプロトコルとして抽象化する。
新しい推論タスクはこのプロトコルを実装すれば Backtester に差し込める。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
import pandas as pd


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

    def __init__(self, threshold: float = 0.0):
        """
        Args:
            threshold: シグナル発生の閾値（予測変化率の絶対値）。
                       0.0 の場合は pred > 0 で buy、pred < 0 で sell。
        """
        self.threshold = threshold

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
        signal[pred > self.threshold] = 1
        signal[pred < -self.threshold] = -1
        return signal
