import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

from typing import Union
def get_stock_data_from_file(market: str, symbol: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
    """
    ローカルCSVから株価データを取得する

    Args:
        market (str): 市場名（例: "us"）
        symbol (str): ティッカー（例: "AAPL"）
        start_date (Union[str, datetime]): データ取得開始日
        end_date (Union[str, datetime]): データ取得終了日

    Returns:
        pd.DataFrame: 株価データ (OHLCV)
    """
    # 例: python/data/us_AAPL/配下の最初のCSVファイルを自動検出
    folder = f"python/data/{market}_{symbol}"
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"CSVファイルが存在しません: {folder}")
    file_path = os.path.join(folder, csv_files[0])
    # 'Date'列が存在しない場合にも対応
    try:
        df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
        # 期間フィルタ
        df = df.loc[start_date:end_date]
    except ValueError:
        # 'Date'列がない場合はそのまま読み込む
        df = pd.read_csv(file_path)
        # 期間フィルタ（Date列があればfilter、なければスキップ）
        if "Date" in df.columns:
            mask = (df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))
            df = df.loc[mask]
    return df

def get_stock_data(
    ticker: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime]
) -> pd.DataFrame:
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
        return data
    # マルチインデックス列をフラット化
    if isinstance(data.columns, pd.MultiIndex):
        # 例: ('Close', 'AAPL') → 'Close'
        data.columns = [str(col[0]) for col in data.columns.values]
    return data

def get_stock_data_auto(
    market: str,
    symbol: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    source: str = "file"
) -> pd.DataFrame:
    """
    APIまたはローカルファイルから株価データを取得するラッパー

    Args:
        market (str): 市場名（例: "us"）
        symbol (str): ティッカー（例: "AAPL"）
        start_date (Union[str, datetime]): データ取得開始日
        end_date (Union[str, datetime]): データ取得終了日
        source (str): "file" または "api"

    Returns:
        pd.DataFrame: 株価データ (OHLCV)
    """
    if source == "file":
        return get_stock_data_from_file(market, symbol, start_date, end_date)
    elif source == "api":
        ticker = symbol if market == "" else f"{symbol}"
        return get_stock_data(ticker, start_date, end_date)
    else:
        raise ValueError(f"sourceは'file'または'api'のみ対応: {source}")

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
