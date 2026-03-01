import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
from src.utils.data_path_utils import get_data_subdir, get_ticker
from src.utils.db import load_stock_features, load_raw_ohlcv
from src.utils.retry_helper import retry_ticker_history, retry_yfinance_download

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
    指定されたマーケット・ティッカーの株価データを取得する（リトライ対応）

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
    
    # リトライロジック付きでデータ取得
    data = retry_ticker_history(
        ticker_obj,
        start=start_date,
        end=end_date_adj,
        auto_adjust=True
    )
    
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
    指定された通貨ペアの為替レートを取得する（リトライ対応）

    Args:
        ticker (str): 通貨ペアのティッカー (例: "JPY=X")
        start_date (Union[str, datetime]): データ取得開始日
        end_date (Union[str, datetime]): データ取得終了日

    Returns:
        pd.DataFrame: 為替レートデータ
    """
    # リトライロジック付きでデータ取得
    data = retry_yfinance_download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )
    
    if data.empty:
        raise ValueError(f"No data found for forex ticker {ticker}")
    return data


def get_latest_business_day(target_date: Optional[datetime] = None) -> datetime:
    """
    指定の日付から遡って、直近の営業日（平日）を返す。
    
    Args:
        target_date: 基準日時（デフォルト: 今日）
    
    Returns:
        営業日（平日）のdatetime
    """
    if target_date is None:
        target_date = datetime.now()
    
    # datetime型に統一
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    
    # 土日を除外して営業日を見つける（最大7日遡る）
    for i in range(7):
        check_date = target_date - timedelta(days=i)
        # 平日（0=月, ...4=金, 5=土, 6=日）
        if check_date.weekday() < 5:
            return check_date
    
    # 万が一見つからない場合は元の日付を返す
    return target_date


def should_fetch_fresh_data(
    db_latest_date: Optional[datetime],
    end_date: Optional[datetime] = None
) -> bool:
    """
    DBの最新日付が古い場合、yfinanceから取得すべきかどうかを判定する。
    
    Args:
        db_latest_date: DB内の最新日付
        end_date: 目標終了日（デフォルト: 今日）
    
    Returns:
        Trueならyfinanceから取得すべき、FalseならDBで十分
    """
    if db_latest_date is None:
        return True  # DB内にデータなし
    
    if end_date is None:
        end_date = datetime.now()
    
    # datetime型に統一
    if isinstance(db_latest_date, str):
        db_latest_date = pd.to_datetime(db_latest_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # あるべき最新営業日を計算
    expected_latest = get_latest_business_day(end_date)
    
    # DB内の日付が期待値より1日以上古い場合は、新規取得が必要
    return (expected_latest - db_latest_date).days >= 1


def merge_market_data(
    db_data: pd.DataFrame,
    fresh_data: pd.DataFrame
) -> pd.DataFrame:
    """
    DB内の既存データと新規に取得したデータを統合する。
    
    Args:
        db_data: DB内のOHLCVデータ（インデックス=Date）
        fresh_data: yfinanceから取得した新規OHLCVデータ（インデックス=Date）
    
    Returns:
        統合されたOHLCVデータ
    """
    if db_data is None or db_data.empty:
        return fresh_data
    
    if fresh_data is None or fresh_data.empty:
        return db_data
    
    # インデックスがDateである確認
    db_data_copy = db_data.copy()
    fresh_data_copy = fresh_data.copy()
    
    # インデックスをタイムゾーン除去（混在を回避）
    # tz_convert(None): tz-aware → tz-naive への正しい変換
    if db_data_copy.index.tz is not None:
        db_data_copy.index = db_data_copy.index.tz_convert(None)
    if fresh_data_copy.index.tz is not None:
        fresh_data_copy.index = fresh_data_copy.index.tz_convert(None)
    
    # 既存データの最新日付より新しいデータのみ取得
    db_latest = pd.to_datetime(db_data_copy.index.max())
    new_rows = fresh_data_copy[fresh_data_copy.index > db_latest]
    
    # 既存データと新規データを結合
    merged = pd.concat([db_data_copy, new_rows]).drop_duplicates().sort_index()
    
    return merged
