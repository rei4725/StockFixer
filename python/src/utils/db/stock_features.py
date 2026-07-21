"""
stock_features テーブルの CRUD 操作

銘柄ごとのテクニカル指標・特徴量データを管理する。
"""

from typing import Optional

import pandas as pd
import psycopg

from src.utils.db._bulk import bulk_insert
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _ensure_columns(con: psycopg.Connection, df: pd.DataFrame) -> None:
    """DataFrame の列が stock_features テーブルに存在しない場合 ALTER TABLE で追加する。

    型推定を行い、適切な SQL 型で列を追加する。
    stock_features は特徴量エンジニアリングで列が頻繁に増えるため、
    migrations一本化の原則の例外として動的ALTERを維持する。
    """
    existing_cols: set = set()
    try:
        result = con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'stock_features'"
        ).fetchall()
        existing_cols = {row[0] for row in result}
    except Exception as e:
        logger.warning(f"stock_features カラム一覧取得失敗: {e}")

    reserved = {"market", "symbol", "row_num"}
    for col in df.columns:
        if col not in existing_cols and col not in reserved:
            dtype = df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                sql_type = "BIGINT"
            elif pd.api.types.is_float_dtype(dtype):
                sql_type = "DOUBLE PRECISION"
            elif pd.api.types.is_bool_dtype(dtype):
                sql_type = "BOOLEAN"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                sql_type = "TIMESTAMP"
            else:
                sql_type = "VARCHAR"
            try:
                con.execute(
                    f'ALTER TABLE stock_features ADD COLUMN IF NOT EXISTS "{col}" {sql_type}'
                )
            except Exception:
                logger.debug(f"カラム追加スキップ（既存）: {col}")


def upsert_stock_features(market: str, symbol: str, df: pd.DataFrame) -> None:
    """
    指定 market/symbol の特徴量データを保存する。
    既存データは DELETE してから INSERT する（べき等）。

    Args:
        market: マーケット識別子 (例: "us", "jp")
        symbol: 銘柄シンボル (例: "AAPL", "7203")
        df: 保存する DataFrame（市場・銘柄の全行）
    """
    save_df = df.copy()
    # DatetimeIndex を date 列として保存する（バックテスト等で日付が必要なため）
    if isinstance(save_df.index, pd.DatetimeIndex) and "date" not in save_df.columns:
        save_df = save_df.reset_index()
        first_col = save_df.columns[0]
        if first_col != "date":
            save_df = save_df.rename(columns={first_col: "date"})
    save_df["market"] = market
    save_df["symbol"] = symbol
    save_df["row_num"] = range(len(save_df))

    with _db_connection() as con:
        _ensure_columns(con, save_df)
        con.execute(
            "DELETE FROM stock_features WHERE market = %s AND symbol = %s", [market, symbol]
        )
        bulk_insert(con, "stock_features", save_df)
    logger.info(f"DB保存完了: stock_features [{market}_{symbol}] ({len(save_df)}行)")


def load_stock_features(market: str, symbol: str) -> Optional[pd.DataFrame]:
    """
    1銘柄分の特徴量を DB から取得する。

    Returns:
        特徴量 DataFrame、データがなければ None
    """
    with _db_connection() as con:
        try:
            df = pd.read_sql(
                "SELECT * FROM stock_features "
                "WHERE market = %(market)s AND symbol = %(symbol)s ORDER BY row_num",
                con,
                params={"market": market, "symbol": symbol},
            )
        except Exception as e:
            logger.error(f"stock_features 読み込み失敗 [{market}_{symbol}]: {e}", exc_info=True)
            return None

    if df.empty:
        return None

    drop_cols = [c for c in ["market", "symbol", "row_num"] if c in df.columns]
    return df.drop(columns=drop_cols)


def load_all_stock_features() -> pd.DataFrame:
    """
    全銘柄の特徴量を DB から取得する（統合モデル学習用）。

    Returns:
        全データを結合した DataFrame（market, symbol 列付き）
    """
    with _db_connection() as con:
        try:
            df = pd.read_sql("SELECT * FROM stock_features ORDER BY market, symbol, row_num", con)
        except Exception as e:
            logger.error(f"stock_features 全件読み込み失敗: {e}", exc_info=True)
            return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    if "row_num" in df.columns:
        df = df.drop(columns=["row_num"])

    logger.info(f"DB読み込み完了: stock_features ({len(df)}行)")
    return df


def delete_stock_features(market: str, symbol: str) -> None:
    """指定 market/symbol のデータを削除する"""
    with _db_connection() as con:
        con.execute(
            "DELETE FROM stock_features WHERE market = %s AND symbol = %s", [market, symbol]
        )
    logger.info(f"DB削除完了: stock_features [{market}_{symbol}]")


def get_all_symbols() -> list:
    """
    stock_features テーブルに存在する全銘柄の (market, symbol) リストを返す。

    Returns:
        list of (market, symbol) tuples
    """
    with _db_connection() as con:
        try:
            result = con.execute(
                "SELECT DISTINCT market, symbol FROM stock_features ORDER BY market, symbol"
            ).fetchall()
            return [(row[0], row[1]) for row in result]
        except Exception as e:
            logger.error(f"stock_features 銘柄一覧取得失敗: {e}", exc_info=True)
            return []
