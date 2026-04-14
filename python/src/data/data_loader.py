from datetime import datetime, timedelta
from typing import Optional, Union

import pandas as pd
import yfinance as yf

from src.utils import yf_client
from src.utils.data_path_utils import get_ticker
from src.utils.db import load_raw_ohlcv, load_stock_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_earnings_dates(market: str, symbol: str, limit: int = 12) -> pd.DatetimeIndex:
    """yfinance から決算日一覧を取得して返す。取得失敗時は空配列。"""
    ticker = get_ticker(market, symbol)
    try:
        ticker_obj = yf.Ticker(ticker)
        getter = getattr(ticker_obj, "get_earnings_dates", None)
        if callable(getter):
            earnings_df = getter(limit=limit)
        else:
            earnings_df = getattr(ticker_obj, "earnings_dates", pd.DataFrame())

        if earnings_df is None or len(earnings_df) == 0:
            return pd.DatetimeIndex([])

        raw_dates = earnings_df.index if isinstance(earnings_df, pd.DataFrame) else earnings_df
        dates = pd.DatetimeIndex(pd.to_datetime(raw_dates, errors="coerce"))
        if dates.tz is not None:
            dates = dates.tz_localize(None)
        dates = dates.dropna().normalize().unique().sort_values()
        return pd.DatetimeIndex(dates)
    except Exception as exc:
        logger.warning(f"決算日取得失敗 [{market}/{symbol}]: {exc}")
        return pd.DatetimeIndex([])


def get_raw_ohlcv_from_db(
    market: str, symbol: str, start_date=None, end_date=None, timeframe: str = "1d"
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


def get_stock_data_from_db(
    market: str,
    symbol: str,
    start_date: Union[str, datetime] = None,
    end_date: Union[str, datetime] = None,
) -> pd.DataFrame:
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
        df = df.loc[pd.to_datetime(start_date) : pd.to_datetime(end_date)]  # type: ignore[misc]

    return df


def get_stock_data_from_file(
    market: str,
    symbol: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
) -> pd.DataFrame:
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
    end_date: Union[str, datetime],
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

    # yfinanceのend_dateはexclusive（その日を含まない）ため、1日加算
    if isinstance(end_date, str):
        end_date_adj = (pd.to_datetime(end_date) + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        end_date_adj = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")

    # リトライロジック付きでデータ取得
    data = yf_client.ticker_history(yf_ticker, start=str(start_date), end=end_date_adj)

    return data


def get_stock_data_auto(
    market: str,
    symbol: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    source: str = "file",
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


def get_forex_data(
    ticker: str, start_date: Union[str, datetime], end_date: Union[str, datetime]
) -> pd.DataFrame:
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
    data = yf_client.download(ticker, start=str(start_date), end=str(end_date), auto_adjust=True)

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
    db_latest_date: Optional[datetime], end_date: Optional[datetime] = None
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


def merge_market_data(db_data: pd.DataFrame, fresh_data: pd.DataFrame) -> pd.DataFrame:
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
    if isinstance(db_data_copy.index, pd.DatetimeIndex) and db_data_copy.index.tz is not None:
        db_data_copy.index = db_data_copy.index.tz_convert(None)
    if isinstance(fresh_data_copy.index, pd.DatetimeIndex) and fresh_data_copy.index.tz is not None:
        fresh_data_copy.index = fresh_data_copy.index.tz_convert(None)

    # 既存データの最新日付より新しいデータのみ取得
    db_latest = pd.to_datetime(db_data_copy.index.max())
    new_rows = fresh_data_copy[fresh_data_copy.index > db_latest]

    # 既存データと新規データを結合
    merged = pd.concat([db_data_copy, new_rows]).drop_duplicates().sort_index()

    return merged


# ---------------------------------------------------------------------------
# クロスアセット特徴量（R-306）
# ---------------------------------------------------------------------------

# 取得するクロスアセットティッカーの定義
_CROSS_ASSET_TICKERS: dict[str, str] = {
    "vix_close": "^VIX",  # 市場恐怖指数
    "usdjpy_close": "JPY=X",  # USD/JPY 為替レート
    "tnx_close": "^TNX",  # 米国10年債利回り
}


def fetch_cross_asset_features(start_date: str, end_date: str) -> "Optional[pd.DataFrame]":
    """
    VIX / USD/JPY / 米国10年債利回りを yfinance から取得し、
    日付インデックス付きの DataFrame として返す（R-306）。

    取得失敗したティッカーは列ごとスキップし、全部失敗した場合のみ None を返す。

    Args:
        start_date: 取得開始日（YYYY-MM-DD）
        end_date: 取得終了日（YYYY-MM-DD）

    Returns:
        カラム: vix_close, usdjpy_close, tnx_close（取得できたもの）
        全部失敗時は None
    """
    import warnings

    frames: list[pd.Series] = []
    for col_name, ticker in _CROSS_ASSET_TICKERS.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hist = yf_client.ticker_history(
                    ticker,
                    start=start_date,
                    end=(pd.to_datetime(end_date) + timedelta(days=1)).strftime("%Y-%m-%d"),
                )
            if hist is None or hist.empty or "Close" not in hist.columns:
                continue
            close = hist["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.rename(col_name)
            # タイムゾーン除去
            close.index = pd.DatetimeIndex(close.index).tz_localize(None)
            frames.append(close)
        except Exception as e:
            logger.debug(f"クロスアセット取得スキップ [{ticker}]: {e}")
            continue

    if not frames:
        return None

    result = pd.concat(frames, axis=1, sort=False)
    result.sort_index(inplace=True)
    return result
