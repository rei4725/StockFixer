"""
DuckDB データベースアクセスモジュール

アプリ全体のDB接続・テーブル操作を一元管理する。
他のモジュールはこのファイルの関数を使用してDB操作を行うこと。
"""

import duckdb
import pandas as pd
from typing import Optional
from threading import Lock

from src.utils.data_path_utils import get_db_path, ensure_dir, get_data_dir

# --- 接続管理（スレッドセーフ） ---
_connection: Optional[duckdb.DuckDBPyConnection] = None
_connection_lock = Lock()


def get_connection() -> duckdb.DuckDBPyConnection:
    """
    DuckDBファイルへのシングルトン接続を返す。
    初回呼び出し時にテーブルを自動初期化する。
    """
    global _connection
    with _connection_lock:
        if _connection is None:
            ensure_dir(get_data_dir())
            db_path = get_db_path()
            _connection = duckdb.connect(db_path)
            _init_tables(_connection)
    return _connection


def close_connection() -> None:
    """DB接続を閉じる"""
    global _connection
    with _connection_lock:
        if _connection is not None:
            _connection.close()
            _connection = None


def get_readonly_connection() -> duckdb.DuckDBPyConnection:
    """
    読み取り専用の新規接続を返す（別プロセスからの利用向け）。
    呼び出し側で close() すること。
    """
    ensure_dir(get_data_dir())
    db_path = get_db_path()
    return duckdb.connect(db_path, read_only=True)


# --- テーブル初期化 ---
def _init_tables(con: duckdb.DuckDBPyConnection) -> None:
    """stock_features / prediction_results テーブルを作成する"""
    con.execute("""
        CREATE TABLE IF NOT EXISTS stock_features (
            market   VARCHAR NOT NULL,
            symbol   VARCHAR NOT NULL,
            row_num  INTEGER NOT NULL,
            PRIMARY KEY (market, symbol, row_num)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS prediction_results (
            run_timestamp  VARCHAR NOT NULL,
            market         VARCHAR NOT NULL,
            symbol         VARCHAR NOT NULL,
            current_price  DOUBLE,
            avg_pred_price DOUBLE,
            diff_ratio     DOUBLE,
            model_count    INTEGER,
            rank_type      VARCHAR,
            PRIMARY KEY (run_timestamp, market, symbol)
        )
    """)


def init_tables() -> None:
    """外部から明示的にテーブル初期化する場合に使用"""
    con = get_connection()
    _init_tables(con)


# --- stock_features テーブル操作 ---

def _ensure_columns(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    """
    DataFrameの列がstock_featuresテーブルに存在しない場合、ALTER TABLEで追加する。
    DuckDBは動的型付けに柔軟なので、全てDOUBLE型で追加する（market/symbol/row_num以外）。
    """
    existing_cols = set()
    try:
        result = con.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'stock_features'").fetchall()
        existing_cols = {row[0] for row in result}
    except Exception:
        pass

    reserved = {"market", "symbol", "row_num"}
    for col in df.columns:
        if col not in existing_cols and col not in reserved:
            # 型推定
            dtype = df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                sql_type = "BIGINT"
            elif pd.api.types.is_float_dtype(dtype):
                sql_type = "DOUBLE"
            elif pd.api.types.is_bool_dtype(dtype):
                sql_type = "BOOLEAN"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                sql_type = "TIMESTAMP"
            else:
                sql_type = "VARCHAR"
            try:
                con.execute(f'ALTER TABLE stock_features ADD COLUMN "{col}" {sql_type}')
            except Exception:
                # 既に存在する場合（レースコンディション対策）
                pass


def upsert_stock_features(market: str, symbol: str, df: pd.DataFrame) -> None:
    """
    指定 market/symbol の特徴量データを保存する。
    既存データは DELETE してから INSERT する（現行CSVの全削除→書き出しと同等）。

    Args:
        market: マーケット識別子 (例: "us", "jp")
        symbol: 銘柄シンボル (例: "AAPL", "7203")
        df: 保存するDataFrame（市場・銘柄の全行）
    """
    con = get_connection()

    # market, symbol 列がDFにあれば除外（テーブル側で管理）
    save_df = df.copy()
    save_df["market"] = market
    save_df["symbol"] = symbol
    save_df["row_num"] = range(len(save_df))

    # テーブルのカラムを必要に応じて追加
    _ensure_columns(con, save_df)

    # 既存データ削除
    con.execute(
        "DELETE FROM stock_features WHERE market = ? AND symbol = ?",
        [market, symbol]
    )

    # INSERT（列名を明示的に指定して順序問題を回避）
    cols = list(save_df.columns)
    col_list = ", ".join(f'"{c}"' for c in cols)
    con.execute(
        f"INSERT INTO stock_features ({col_list}) SELECT {col_list} FROM save_df"
    )
    print(f"DB保存完了: stock_features [{market}_{symbol}] ({len(save_df)}行)")


def load_stock_features(market: str, symbol: str) -> Optional[pd.DataFrame]:
    """
    1銘柄分の特徴量をDBから取得する。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル

    Returns:
        特徴量DataFrame、データが無ければNone
    """
    con = get_connection()
    try:
        df = con.execute(
            "SELECT * FROM stock_features WHERE market = ? AND symbol = ? ORDER BY row_num",
            [market, symbol]
        ).fetchdf()
    except Exception:
        return None

    if df.empty:
        return None

    # 管理列を除外
    drop_cols = [c for c in ["market", "symbol", "row_num"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df


def load_all_stock_features() -> pd.DataFrame:
    """
    全銘柄の特徴量をDBから取得する（統合モデル学習用）。

    Returns:
        全データを結合したDataFrame（market, symbol列付き）
    """
    con = get_connection()
    try:
        df = con.execute(
            "SELECT * FROM stock_features ORDER BY market, symbol, row_num"
        ).fetchdf()
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # row_numは不要
    if "row_num" in df.columns:
        df = df.drop(columns=["row_num"])

    print(f"DB読み込み完了: stock_features ({len(df)}行)")
    return df


def delete_stock_features(market: str, symbol: str) -> None:
    """指定 market/symbol のデータを削除する"""
    con = get_connection()
    con.execute(
        "DELETE FROM stock_features WHERE market = ? AND symbol = ?",
        [market, symbol]
    )
    print(f"DB削除完了: stock_features [{market}_{symbol}]")


def get_all_symbols() -> list:
    """
    stock_featuresテーブルに存在する全銘柄の (market, symbol) リストを返す

    Returns:
        list of (market, symbol) tuples
    """
    con = get_connection()
    try:
        result = con.execute(
            "SELECT DISTINCT market, symbol FROM stock_features ORDER BY market, symbol"
        ).fetchall()
        return [(row[0], row[1]) for row in result]
    except Exception:
        return []


# --- prediction_results テーブル操作 ---

def save_prediction_results(run_timestamp: str, df: pd.DataFrame, rank_type: str = None) -> None:
    """
    予測結果をDBに保存する。

    Args:
        run_timestamp: 実行タイムスタンプ (例: "20260213_142903")
        df: 予測結果DataFrame (market, symbol, current_price, avg_pred_price, diff_ratio, model_count)
        rank_type: 'top10' / 'worst10' / None
    """
    con = get_connection()

    save_df = df.copy()
    save_df["run_timestamp"] = run_timestamp
    if rank_type is not None:
        save_df["rank_type"] = rank_type
    elif "rank_type" not in save_df.columns:
        save_df["rank_type"] = None

    # 必要な列のみ選択
    cols = ["run_timestamp", "market", "symbol", "current_price", "avg_pred_price", "diff_ratio", "model_count", "rank_type"]
    save_df = save_df[[c for c in cols if c in save_df.columns]]

    # 不足列補完
    for c in cols:
        if c not in save_df.columns:
            save_df[c] = None

    con.execute("INSERT INTO prediction_results SELECT * FROM save_df")
    print(f"DB保存完了: prediction_results [{run_timestamp}] ({len(save_df)}行)")


def load_prediction_results(
    run_timestamp: str = None,
    market: str = None,
    rank_type: str = None
) -> pd.DataFrame:
    """
    予測結果をDBから取得する。

    Args:
        run_timestamp: 実行タイムスタンプ（Noneなら最新）
        market: マーケットフィルタ（Noneなら全マーケット）
        rank_type: 'top10' / 'worst10'（Noneなら全件）

    Returns:
        予測結果DataFrame
    """
    con = get_connection()

    if run_timestamp is None:
        run_timestamp = load_latest_prediction_timestamp()
        if run_timestamp is None:
            return pd.DataFrame()

    query = "SELECT * FROM prediction_results WHERE run_timestamp = ?"
    params = [run_timestamp]

    if market is not None:
        query += " AND market = ?"
        params.append(market)

    if rank_type is not None:
        query += " AND rank_type = ?"
        params.append(rank_type)

    query += " ORDER BY diff_ratio DESC"

    try:
        df = con.execute(query, params).fetchdf()
        return df
    except Exception:
        return pd.DataFrame()


def load_latest_prediction_timestamp() -> Optional[str]:
    """
    最新の run_timestamp を返す。

    Returns:
        最新のタイムスタンプ文字列、なければNone
    """
    con = get_connection()
    try:
        result = con.execute(
            "SELECT DISTINCT run_timestamp FROM prediction_results ORDER BY run_timestamp DESC LIMIT 1"
        ).fetchone()
        if result:
            return result[0]
        return None
    except Exception:
        return None


def load_prediction_markets(run_timestamp: str = None) -> list:
    """
    指定タイムスタンプの予測結果に含まれるマーケット一覧を返す。

    Args:
        run_timestamp: Noneなら最新

    Returns:
        マーケット名のリスト
    """
    con = get_connection()

    if run_timestamp is None:
        run_timestamp = load_latest_prediction_timestamp()
        if run_timestamp is None:
            return []

    try:
        result = con.execute(
            "SELECT DISTINCT market FROM prediction_results WHERE run_timestamp = ? ORDER BY market",
            [run_timestamp]
        ).fetchall()
        return [row[0] for row in result]
    except Exception:
        return []
