"""
データ保存モジュール

生の株価データをDBに保存する機能を提供する
特徴量生成を含む処理は src.services.data_pipeline を使用すること
"""

from datetime import datetime
from typing import Optional

import pandas as pd

from src.data.data_loader import get_stock_data
from src.utils.data_path_utils import get_ticker
from src.utils.db import upsert_raw_ohlcv, upsert_stock_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_raw_ohlcv(
    market: str, symbol: str, df: pd.DataFrame, timeframe: str = "1d", source: str = "yfinance"
) -> int:
    """
    yfinanceから取得したOHLCV DataFrameをmarket_data_rawに保存する。
    カラム名の正規化（先頭大文字 → 小文字）を行ってから保存する。

    Args:
        market: マーケット識別子 (例: "us", "jp")
        symbol: 銘柄シンボル (例: "AAPL", "7203")
        df: yfinance形式のOHLCV DataFrame (インデックスは日付、カラムはOpen/High/Low/Close/Volume等)
        timeframe: 時間軸 (default: "1d")
        source: 取得元 (default: "yfinance")

    Returns:
        保存した行数
    """
    ticker = get_ticker(market, symbol)
    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Adj Close": "adj_close",
        "Dividends": None,
        "Stock Splits": None,
    }
    rows = []
    for ts, row in df.iterrows():
        rec: dict = {
            "market": market,
            "symbol": symbol,
            "ticker": ticker,
            "timeframe": timeframe,
            "ts": (
                pd.Timestamp(ts).tz_localize(None)  # type: ignore[arg-type]
                if pd.Timestamp(ts).tzinfo  # type: ignore[arg-type]
                else pd.Timestamp(ts)  # type: ignore[arg-type]
            ),
            "source": source,
        }
        for src_col, dst_col in col_map.items():
            if dst_col is None:
                continue
            if src_col in row.index:
                val = row[src_col]
                # NaNは保存しない（adj_closeなどauto_adjust=True時に不要な列を除外）
                if pd.notna(val):
                    rec[dst_col] = val
            elif src_col.lower() in [c.lower() for c in row.index]:
                # 大文字小文字ゆらぎ吸収
                matched = next(c for c in row.index if c.lower() == src_col.lower())
                val = row[matched]
                if pd.notna(val):
                    rec[dst_col] = val
        rows.append(rec)
    return upsert_raw_ohlcv(rows)


def save_raw_stock_data(
    market: str,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    out_dir: str = None,  # 後方互換のため残置（未使用）
) -> Optional[pd.DataFrame]:
    """
    指定した市場・シンボル・期間の生の株価データを取得し、DBに保存する。
    特徴量生成は行わない（data層の責務として純粋なデータ保存のみ）

    Args:
        market: マーケット識別子 (例: "us", "jp")
        symbol: 銘柄シンボル (例: "AAPL", "7203")
        start_date: 開始日 (省略時は取得可能な最古日)
        end_date: 終了日 (省略時は現在日)
        out_dir: 後方互換のため残置（未使用）

    Returns:
        保存したDataFrame、または取得失敗時はNone
    """
    # start_date, end_date自動決定
    if start_date is None or end_date is None:
        try:
            df_all = get_stock_data(
                market, symbol, "1900-01-01", datetime.now().strftime("%Y-%m-%d")
            )
            if df_all is None or df_all.empty:
                logger.warning("データが取得できませんでした: symbol=%s", symbol)
                return None
            start_date = df_all.index.min().strftime("%Y-%m-%d")
        except Exception:
            logger.error("データ取得エラー: symbol=%s", symbol, exc_info=True)
            return None
        end_date = datetime.now().strftime("%Y-%m-%d")

    # 市場ごとにティッカーを補正
    ticker = get_ticker(market, symbol)

    logger.info(
        "データ取得: market=%s, symbol=%s, ticker=%s, %s～%s",
        market,
        symbol,
        ticker,
        start_date,
        end_date,
    )
    df = get_stock_data(market, ticker, start_date, end_date)
    if df is None or df.empty:
        logger.warning("データが取得できませんでした: market=%s, symbol=%s", market, symbol)
        return None

    # DBに保存
    upsert_stock_features(market, symbol, df)

    return df
