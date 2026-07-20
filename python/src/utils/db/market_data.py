"""
market_data_raw テーブルの CRUD 操作

生OHLCVデータ（yfinance 等から取得した株価データ）を管理する。
"""

from typing import Optional

import pandas as pd

from src.utils.db._bulk import bulk_upsert
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def upsert_raw_ohlcv(rows: list[dict]) -> int:
    """
    生OHLCVデータを market_data_raw テーブルに UPSERT 保存する。
    同一 (market, symbol, timeframe, ts) は上書きされる（べき等）。

    Args:
        rows: 各行を表す辞書のリスト。必須キー: market, symbol, ticker, timeframe, ts,
              open, high, low, close, volume。任意: adj_close, source

    Returns:
        保存した行数
    """
    if not rows:
        return 0

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    if "adj_close" not in df.columns:
        df["adj_close"] = None
    if "source" not in df.columns:
        df["source"] = "yfinance"
    df["ingested_at"] = pd.Timestamp.now("UTC")
    if "volume" in df.columns:
        # 呼び出し元（saver.py の DataFrame.iterrows()）は価格列と出来高列が
        # 混在する行を反復するため、Volume が本来 int でも float64 へ
        # 暗黙アップキャストされることがある。market_data_raw.volume は
        # BIGINT のため、float のまま COPY すると "1234.0" のようなテキスト
        # 表現になり Postgres が拒否する。ここで明示的に整数へ丸める。
        df["volume"] = df["volume"].astype("Int64")

    cols = [
        "market",
        "symbol",
        "ticker",
        "timeframe",
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adj_close",
        "source",
        "ingested_at",
    ]
    df = df[[c for c in cols if c in df.columns]]

    with _db_connection() as con:
        bulk_upsert(
            con,
            "market_data_raw",
            df,
            key_cols=["market", "symbol", "timeframe", "ts"],
            columns=[
                "market",
                "symbol",
                "ticker",
                "timeframe",
                "ts",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "adj_close",
                "source",
                "ingested_at",
            ],
        )
    n = len(df)
    logger.info(
        f"DB保存完了: market_data_raw "
        f"[{rows[0].get('market', '')}_{rows[0].get('symbol', '')}] ({n}行)"
    )
    return n


def load_all_raw_ohlcv_symbols(timeframe: str = "1d") -> list:
    """
    market_data_raw テーブルに存在する全 (market, symbol) のリストを返す。

    Args:
        timeframe: 時間軸 (default: "1d")

    Returns:
        [(market, symbol), ...] のリスト
    """
    with _db_connection() as con:
        try:
            rows = con.execute(
                "SELECT DISTINCT market, symbol FROM market_data_raw"
                " WHERE timeframe = %s ORDER BY market, symbol",
                [timeframe],
            ).fetchall()
            return [(r[0], r[1]) for r in rows]
        except Exception as e:
            logger.error(f"load_all_raw_ohlcv_symbols 失敗: {e}", exc_info=True)
            return []


def load_raw_ohlcv(
    market: str, symbol: str, start_date=None, end_date=None, timeframe: str = "1d"
) -> Optional[pd.DataFrame]:
    """
    market_data_raw テーブルから生OHLCVを取得する。

    Args:
        market: マーケット識別子 (例: "us", "jp")
        symbol: 銘柄シンボル (例: "AAPL", "7203")
        start_date: 開始日 (str or datetime, None なら全期間)
        end_date: 終了日 (str or datetime, None なら全期間)
        timeframe: 時間軸 (default: "1d")

    Returns:
        OHLCVのDataFrame (インデックス=ts)、データなしは None
    """
    query = """
        SELECT ts, open, high, low, close, volume, adj_close
        FROM market_data_raw
        WHERE market = %s AND symbol = %s AND timeframe = %s
    """
    params: list = [market, symbol, timeframe]

    if start_date is not None:
        query += " AND ts >= %s"
        params.append(pd.to_datetime(start_date))
    if end_date is not None:
        query += " AND ts <= %s"
        params.append(pd.to_datetime(end_date))

    query += " ORDER BY ts"

    with _db_connection() as con:
        try:
            df = pd.read_sql(query, con, params=params)
        except Exception as e:
            logger.error(f"market_data_raw 読み込み失敗 [{market}_{symbol}]: {e}", exc_info=True)
            return None

    if df.empty:
        return None

    df = df.set_index("ts")
    df.index.name = "Date"
    # カラム名を yfinance 形式（先頭大文字）に揃える
    df.columns = pd.Index([c.capitalize() if c != "adj_close" else "Adj Close" for c in df.columns])
    return df
