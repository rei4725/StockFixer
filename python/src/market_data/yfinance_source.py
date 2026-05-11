"""
YFinanceDataSource — yfinance を使ったデフォルトのデータソース実装。

DataSourceBase を継承し、既存の yf_client を内部で使用する。
loader.py の get_stock_data / get_forex_data と同等のロジックを提供する。
"""

from datetime import timedelta

import pandas as pd

import src.market_data.yf_client as yf_client
from src.market_data.base import DataSourceBase
from src.utils.data_path_utils import get_ticker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class YFinanceDataSource(DataSourceBase):
    """yfinance を使ったデフォルトのデータソース実装。"""

    def get_stock_data(
        self,
        symbol: str,
        market: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        yf_ticker = get_ticker(market, symbol)
        # yfinance の end は exclusive なので 1 日加算
        end_date_adj = (pd.to_datetime(end_date) + timedelta(days=1)).strftime("%Y-%m-%d")
        return yf_client.ticker_history(yf_ticker, start=str(start_date), end=end_date_adj)

    def get_forex_data(
        self,
        pair: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        data = yf_client.download(pair, start=str(start_date), end=str(end_date), auto_adjust=True)
        if data.empty:
            raise ValueError(f"No data found for forex pair {pair}")
        return data
