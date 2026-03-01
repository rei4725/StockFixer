import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
from src.utils.data_path_utils import get_data_subdir, get_ticker
from src.utils.db import load_stock_features, load_raw_ohlcv

from typing import Optional, Union


def get_raw_ohlcv_from_db(
    market: str,
    symbol: str,
    start_date=None,
    end_date=None,
    timeframe: str = "1d"
) -> Optional[pd.DataFrame]:
    """
    market_data_rawテーブルから生OHLCVを取得する。
    DBに存在しない場合は None を返す（呼び出し元でyfinanceフォールバックを行う）。

    Args:
        market: マーケット識別子 (例: "us", "jp")
        symbol: 銘柄シンボル (例: "AAPL", "7203")
        start_date: 開始日 (str or datetime)
        end_date: 終了日 (str or datetime)
        timeframe: 時間軸 (default: "1d")

    Returns:
        OHLCVのDataFrame (インデックス=Date)、なければNone
    """
    return load_raw_ohlcv(market, symbol, start_date, end_date, timeframe)


def get_stock_data_from_db(market: str, symbol: str, start_date: Union[str, datetime] = None, end_date: Union[str, datetime] = None) -> pd.DataFrame:
    """
    DuckDB から株価データ（特徴量含む）を取得する

    Args:
        market (str): 市場名（例: "us"）
        symbol (str): ティッカー（例: "AAPL"）
        start_date (Union[str, datetime], optional): データ取得開始日
        end_date (Union[str, datetime], optional): データ取得終了日

    Returns:
        pd.DataFrame: 株価データ
    """
    df = load_stock_features(market, symbol)
    if df is None or df.empty:
        raise FileNotFoundError(f"DBにデータが存在しません: {market}_{symbol}")

    # Date列があれば期間フィルタ
    if "Date" in df.columns and start_date is not None and end_date is not None:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        df = df.loc[start_date:end_date]

    return df


def get_stock_data_from_file(market: str, symbol: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
    """
    DBから株価データを取得する（後方互換のため関数名を維持）

    Args:
        market (str): 市場名（例: "us"）
        symbol (str): ティッカー（例: "AAPL"）
        start_date (Union[str, datetime]): データ取得開始日
        end_date (Union[str, datetime]): データ取得終了日

    Returns:
        pd.DataFrame: 株価データ (OHLCV)
    """
    return get_stock_data_from_db(market, symbol, start_date, end_date)

def get_stock_data(
    market: str,
    ticker: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime]
) -> pd.DataFrame:
    """
    指定されたマーケット・ティッカーの株価データを取得する

    Args:
        market (str): 市場名（例: "us", "jp"）
        ticker (str): ティッカーシンボル (例: "AAPL", "6146")
        start_date (Union[str, datetime]): データ取得開始日 (YYYY-MM-DD)
        end_date (Union[str, datetime]): データ取得終了日 (YYYY-MM-DD)

    Returns:
        pd.DataFrame: 株価データ (OHLCV)
    """
    # マーケットに応じてtickerを変換
    yf_ticker = get_ticker(market, ticker)
    
    # yf.Ticker().history() を使用（スレッドセーフ）
    # yf.download() は並列実行時にデータが混ざる可能性があるため非推奨
    ticker_obj = yf.Ticker(yf_ticker)
    
    # yfinanceのend_dateはexclusive（その日を含まない）ため、1日加算
    if isinstance(end_date, str):
        end_date_adj = (pd.to_datetime(end_date) + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        end_date_adj = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")
    
    data = ticker_obj.history(start=start_date, end=end_date_adj, auto_adjust=True)
    
    if data.empty:
        return data
    # マルチインデックス列をフラット化
    if isinstance(data.columns, pd.MultiIndex):
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
        ticker = get_ticker(market, symbol)
        return get_stock_data(market, ticker, start_date, end_date)
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
