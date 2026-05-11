"""
DataSourceBase — データソース抽象基底クラス

BrokerBase と同様のパターンで、yfinance 以外のデータソース（Bloomberg 等）への
差し替えやテスト時のモック注入を可能にする。
"""

from abc import ABC, abstractmethod

import pandas as pd


class DataSourceBase(ABC):
    """株価・為替データ取得のインターフェース。実装はデータソースごとに行う。"""

    @abstractmethod
    def get_stock_data(
        self,
        symbol: str,
        market: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        株価 OHLCV データを取得して返す。

        Args:
            symbol: 銘柄シンボル（例: "AAPL", "7203"）
            market: マーケット識別子（例: "us", "jp"）
            start_date: 開始日 (YYYY-MM-DD)
            end_date: 終了日 (YYYY-MM-DD)

        Returns:
            OHLCV DataFrame。取得失敗時は空 DataFrame を返す。
        """

    @abstractmethod
    def get_forex_data(
        self,
        pair: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        為替レートデータを取得して返す。

        Args:
            pair: 通貨ペアのティッカー（例: "JPY=X"）
            start_date: 開始日 (YYYY-MM-DD)
            end_date: 終了日 (YYYY-MM-DD)

        Returns:
            為替レート DataFrame。取得失敗時は空 DataFrame を返す。
        """
