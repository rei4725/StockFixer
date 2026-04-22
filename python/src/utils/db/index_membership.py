"""
index_membership_history テーブルの CRUD 操作

指数構成銘柄（S&P500 / 日経225 など）のスナップショット履歴を管理する。
"""

from datetime import date

import pandas as pd

from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

INDEX_NAME_BY_MARKET = {
    "us": "sp500",
    "jp": "nikkei225",
}


def _normalize_symbol(symbol: str, market: str) -> str:
    value = str(symbol).strip()
    if not value:
        return ""
    if market == "us":
        return value.upper()
    return value


def save_index_membership_snapshot(
    market: str,
    symbols: list[str],
    snapshot_date: str | date | None = None,
    source: str = "wikipedia",
) -> int:
    """
    指数構成銘柄スナップショットを保存する（同一キーは上書き）。

    Args:
        market: マーケット識別子
        symbols: 銘柄コード一覧
        snapshot_date: 有効日。None の場合は当日(UTC)
        source: データソース識別子

    Returns:
        保存行数
    """
    if not symbols:
        return 0

    normalized_market = market.lower()
    unique_symbols = sorted({_normalize_symbol(s, normalized_market) for s in symbols})
    unique_symbols = [s for s in unique_symbols if s]
    if not unique_symbols:
        return 0

    if snapshot_date is None:
        snapshot_day = date.today()
    else:
        snapshot_day = pd.to_datetime(snapshot_date).date()

    df = pd.DataFrame(
        {
            "market": normalized_market,
            "symbol": unique_symbols,
            "index_name": INDEX_NAME_BY_MARKET.get(normalized_market, normalized_market),
            "snapshot_date": snapshot_day,
            "source": source,
            "fetched_at": pd.Timestamp.now("UTC"),
        }
    )

    with _db_connection() as con:
        con.register("_index_membership_temp", df)
        con.execute(
            """
            INSERT OR REPLACE INTO index_membership_history
                (market, symbol, index_name, snapshot_date, source, fetched_at)
            SELECT market, symbol, index_name, snapshot_date, source, fetched_at
            FROM _index_membership_temp
            """
        )

    logger.info(
        f"DB保存完了: index_membership_history [{normalized_market}] "
        f"snapshot_date={snapshot_day.isoformat()} ({len(df)}行)"
    )
    return len(df)


def load_index_membership_symbols_as_of(as_of_date: str | date) -> list[tuple[str, str]]:
    """
    指定日以前で最新の指数構成銘柄スナップショットを読み込む。

    Args:
        as_of_date: 基準日（YYYY-MM-DD）

    Returns:
        (market, symbol) のリスト
    """
    as_of_day = pd.to_datetime(as_of_date).date()

    query = """
        WITH latest_snapshot AS (
            SELECT
                market,
                MAX(snapshot_date) AS snapshot_date
            FROM index_membership_history
            WHERE snapshot_date <= ?
            GROUP BY market
        )
        SELECT h.market, h.symbol
        FROM index_membership_history h
        JOIN latest_snapshot s
          ON h.market = s.market
         AND h.snapshot_date = s.snapshot_date
        ORDER BY h.market, h.symbol
    """
    with _db_connection() as con:
        rows = con.execute(query, [as_of_day]).fetchall()
    return [(row[0], row[1]) for row in rows]
