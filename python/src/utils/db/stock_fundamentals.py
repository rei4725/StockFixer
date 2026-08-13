"""
stock_fundamentals テーブルの read/write 操作。

yfinance から取得した財務スナップショット（1銘柄1行・最新値のみ）を管理する。
ポイントインタイム非対応のため、ライブスクリーン専用。詳細は
``src.market_data.types.FundamentalRecord`` の docstring を参照。

レイヤー規約（utils < BC）により本モジュールは market_data BC の型を import せず、
``_FundamentalRecordLike`` Protocol で record を受け取る（duck typing）。

DuckDB 並列書き込み禁止（CLAUDE.md）。書き込みは ``_db_connection`` 経由で逐次行う。
"""

from datetime import datetime
from typing import Optional, Protocol

import pandas as pd

from src.utils.db._connection import _db_connection
from src.utils.db._read import coerce_object_numeric_columns
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 数値化しない列（as_of は timestamp、market/symbol は文字列）
_NON_NUMERIC_COLUMNS = {"as_of", "market", "symbol"}

# stock_fundamentals の保存対象カラム（DDL と一致させること）
_COLUMNS = [
    "market",
    "symbol",
    "as_of",
    "revenue",
    "operating_income",
    "net_income",
    "eps",
    "roe",
    "op_margin",
    "net_margin",
    "debt_to_equity",
    "cash",
    "market_cap",
    "shares_outstanding",
    "revenue_cagr_3y",
    "trailing_pe",
    "payout_ratio",
]


class _FundamentalRecordLike(Protocol):
    """upsert が必要とする財務レコードの構造（market_data.FundamentalRecord 互換）。"""

    market: str
    symbol: str
    as_of: datetime
    revenue: Optional[float]
    operating_income: Optional[float]
    net_income: Optional[float]
    eps: Optional[float]
    roe: Optional[float]
    op_margin: Optional[float]
    net_margin: Optional[float]
    debt_to_equity: Optional[float]
    cash: Optional[float]
    market_cap: Optional[float]
    shares_outstanding: Optional[float]
    revenue_cagr_3y: Optional[float]
    trailing_pe: Optional[float]
    payout_ratio: Optional[float]


def upsert_fundamentals(record: _FundamentalRecordLike) -> None:
    """
    1銘柄の財務スナップショットを保存する。

    既存の (market, symbol) 行は DELETE してから INSERT する（べき等）。

    Args:
        record: 保存する財務レコード
    """
    market = record.market
    symbol = record.symbol
    values = [getattr(record, name) for name in _COLUMNS]
    col_list = ", ".join(_COLUMNS)
    placeholders = ", ".join("%s" for _ in _COLUMNS)

    with _db_connection() as con:
        con.execute(
            "DELETE FROM stock_fundamentals WHERE market = %s AND symbol = %s",
            [market, symbol],
        )
        con.execute(
            f"INSERT INTO stock_fundamentals ({col_list}) VALUES ({placeholders})",
            values,
        )
    logger.info(f"DB保存完了: stock_fundamentals [{market}_{symbol}]")


def load_fundamentals(market: str, symbol: str) -> Optional[dict]:
    """
    1銘柄分の財務スナップショットを取得する。

    Returns:
        財務フィールドの dict、データがなければ None
    """
    with _db_connection() as con:
        try:
            df = pd.read_sql(
                "SELECT * FROM stock_fundamentals WHERE market = %s AND symbol = %s",
                con,
                params=[market, symbol],
            )
        except Exception as e:
            logger.error(f"stock_fundamentals 読み込み失敗 [{market}_{symbol}]: {e}", exc_info=True)
            return None

    if df.empty:
        return None
    df = coerce_object_numeric_columns(df, exclude=_NON_NUMERIC_COLUMNS)
    return df.iloc[0].to_dict()


def load_all_fundamentals() -> pd.DataFrame:
    """
    全銘柄の財務スナップショットを取得する（ライブスクリーン用）。

    Returns:
        全レコードの DataFrame。データがなければ空 DataFrame
    """
    with _db_connection() as con:
        try:
            df = pd.read_sql("SELECT * FROM stock_fundamentals ORDER BY market, symbol", con)
        except Exception as e:
            logger.error(f"stock_fundamentals 全件読み込み失敗: {e}", exc_info=True)
            return pd.DataFrame()

    if df.empty:
        return df
    return coerce_object_numeric_columns(df, exclude=_NON_NUMERIC_COLUMNS)
