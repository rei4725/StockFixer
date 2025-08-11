import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

from typing import Union

def get_stock_data(ticker: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
    """
    指定されたティッカーの株価データを取得する

    Args:
        ticker (str): ティッカーシンボル (例: "AAPL")
        start_date (Union[str, datetime]): データ取得開始日 (YYYY-MM-DD)
        end_date (Union[str, datetime]): データ取得終了日 (YYYY-MM-DD)

    Returns:
        pd.DataFrame: 株価データ (OHLCV)
    """
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    return data

def get_forex_data(ticker: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
    """
    指定された通貨ペアの為替レートを取得する

    Args:
        ticker (str): 通貨ペアのティッカー (例: "JPY=X")
        start_date (Union[str, datetime]): データ取得開始日
        end_date (Union[str, datetime]): データ取得終了日

    Returns:
        pd.DataFrame: 為替レートデータ
    """
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data found for forex ticker {ticker}")
    return data
