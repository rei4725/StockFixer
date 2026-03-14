"""
DuckDB データベースアクセスモジュール

アプリ全体のDB接続・テーブル操作を一元管理する。
他のモジュールはこのファイルの関数を使用してDB操作を行うこと。
"""

from threading import Lock
from typing import Optional

import duckdb
import pandas as pd
from src.utils.data_path_utils import ensure_dir, get_data_dir, get_db_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

# --- 接続管理（スレッドセーフ） ---
_connection: Optional[duckdb.DuckDBPyConnection] = None
_connection_lock = Lock()


def get_connection() -> duckdb.DuckDBPyConnection:
    """
    DuckDBファイルへのシングルトン接続を返す。
    初回呼び出し時にテーブルを自動初期化する。

    読み取り性能を最適化するため、スレッド数を設定しロック競合を最小化する。
    DuckDB は自動的にロック機構を最適化しており、複数読み取り接続を効率的に処理する。
    """
    global _connection
    with _connection_lock:
        if _connection is None:
            ensure_dir(get_data_dir())
            db_path = get_db_path()

            # DuckDB 接続設定（ロック競合最小化）
            # threads: CPU並列処理によるスループット向上
            # max_memory: メモリ使用量制限で安定性確保
            config = {"threads": "4", "memory_limit": "2GB"}
            _connection = duckdb.connect(db_path, config=config)

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
    """stock_features / prediction_results / market_data_raw テーブルを作成する"""
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_features (
            market   VARCHAR NOT NULL,
            symbol   VARCHAR NOT NULL,
            row_num  INTEGER NOT NULL,
            PRIMARY KEY (market, symbol, row_num)
        )
    """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_results (
            market         VARCHAR NOT NULL,
            symbol         VARCHAR NOT NULL,
            predicted_at   VARCHAR NOT NULL,
            current_price  DOUBLE,
            avg_pred_price DOUBLE,
            diff_ratio     DOUBLE,
            model_count    INTEGER,
            PRIMARY KEY (market, symbol, predicted_at)
        )
    """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS market_data_raw (
            market      VARCHAR NOT NULL,
            symbol      VARCHAR NOT NULL,
            ticker      VARCHAR NOT NULL,
            timeframe   VARCHAR NOT NULL,
            ts          TIMESTAMP NOT NULL,
            open        DOUBLE,
            high        DOUBLE,
            low         DOUBLE,
            close       DOUBLE,
            volume      BIGINT,
            adj_close   DOUBLE,
            source      VARCHAR NOT NULL DEFAULT 'yfinance',
            ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (market, symbol, timeframe, ts)
        )
    """
    )


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
        result = con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'stock_features'"
        ).fetchall()
        existing_cols = {row[0] for row in result}
    except Exception as e:
        logger.warning(f"stock_featuresカラム一覧取得失敗: {e}")

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
                # 既に存在する場合（レースコンディション対策）は無視
                logger.debug(f"カラム追加スキップ（既存）: {col}")


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
    con.execute("DELETE FROM stock_features WHERE market = ? AND symbol = ?", [market, symbol])

    # INSERT（列名を明示的に指定して順序問題を回避）
    cols = list(save_df.columns)
    col_list = ", ".join(f'"{c}"' for c in cols)
    con.execute(f"INSERT INTO stock_features ({col_list}) SELECT {col_list} FROM save_df")
    logger.info(f"DB保存完了: stock_features [{market}_{symbol}] ({len(save_df)}行)")


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
            [market, symbol],
        ).fetchdf()
    except Exception as e:
        logger.error(f"stock_features読み込み失敗 [{market}_{symbol}]: {e}", exc_info=True)
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
        df = con.execute("SELECT * FROM stock_features ORDER BY market, symbol, row_num").fetchdf()
    except Exception as e:
        logger.error(f"stock_features全件読み込み失敗: {e}", exc_info=True)
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # row_numは不要
    if "row_num" in df.columns:
        df = df.drop(columns=["row_num"])

    logger.info(f"DB読み込み完了: stock_features ({len(df)}行)")
    return df


def delete_stock_features(market: str, symbol: str) -> None:
    """指定 market/symbol のデータを削除する"""
    con = get_connection()
    con.execute("DELETE FROM stock_features WHERE market = ? AND symbol = ?", [market, symbol])
    logger.info(f"DB削除完了: stock_features [{market}_{symbol}]")


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
    except Exception as e:
        logger.error(f"stock_features銘柄一覧取得失敗: {e}", exc_info=True)
        return []


# --- prediction_results テーブル操作 ---


def save_prediction_results(predicted_at: str, df: pd.DataFrame) -> None:
    """
    予測結果をDBに保存する（Delete-Insert方式）。
    対象銘柄の既存データを削除してから挿入する。

    Args:
        predicted_at: 予測実行日時 (例: "20260213_142903")
        df: 予測結果DataFrame (market, symbol, current_price, avg_pred_price, diff_ratio, model_count)
    """
    con = get_connection()

    save_df = df.copy()
    save_df["predicted_at"] = predicted_at

    # 必要な列のみ選択
    cols = [
        "market",
        "symbol",
        "predicted_at",
        "current_price",
        "avg_pred_price",
        "diff_ratio",
        "model_count",
    ]
    save_df = save_df[[c for c in cols if c in save_df.columns]]

    # 不足列補完
    for c in cols:
        if c not in save_df.columns:
            save_df[c] = None

    # Delete-Insert: 対象銘柄の既存データを削除
    pairs = save_df[["market", "symbol"]].drop_duplicates()
    for _, row in pairs.iterrows():
        con.execute(
            "DELETE FROM prediction_results WHERE market = ? AND symbol = ?",
            [row["market"], row["symbol"]],
        )

    # DataFrame を登録してから INSERT（複数行の INSERT に対応）
    con.register("_save_df_temp", save_df)
    con.execute("INSERT INTO prediction_results SELECT * FROM _save_df_temp")
    logger.info(f"DB保存完了: prediction_results [{predicted_at}] ({len(save_df)}行)")


def load_prediction_results(
    predicted_at: str = None, market: str = None, top_n: int = None, worst_n: int = None
) -> pd.DataFrame:
    """
    予測結果をDBから取得する。

    Args:
        predicted_at: 予測日時（Noneなら最新）
        market: マーケットフィルタ（Noneなら全マーケット）
        top_n: 上位N件のみ取得（diff_ratio降順）
        worst_n: 下位N件のみ取得（diff_ratio昇順）

    Returns:
        予測結果DataFrame
    """
    con = get_connection()

    if predicted_at is None:
        predicted_at = load_latest_prediction_timestamp()
        if predicted_at is None:
            return pd.DataFrame()

    query = "SELECT * FROM prediction_results WHERE predicted_at = ?"
    params: list = [predicted_at]

    if market is not None:
        query += " AND market = ?"
        params.append(market)

    if worst_n is not None:
        query += " ORDER BY diff_ratio ASC LIMIT ?"
        params.append(worst_n)
    elif top_n is not None:
        query += " ORDER BY diff_ratio DESC LIMIT ?"
        params.append(top_n)
    else:
        query += " ORDER BY diff_ratio DESC"

    try:
        df = con.execute(query, params).fetchdf()
        return df
    except Exception as e:
        logger.error(f"prediction_results読み込み失敗: {e}", exc_info=True)
        return pd.DataFrame()


def load_latest_prediction_timestamp() -> Optional[str]:
    """
    最新の run_timestamp を返す。

    Returns:
        最新のタイムスタンプ文字列、なければNone
    """
    con = get_connection()
    try:
        query = (
            "SELECT DISTINCT predicted_at "
            "FROM prediction_results "
            "ORDER BY predicted_at DESC LIMIT 1"
        )
        result = con.execute(query).fetchone()
        if result:
            return result[0]
        return None
    except Exception as e:
        logger.error(f"最新予測タイムスタンプ取得失敗: {e}", exc_info=True)
        return None


def load_prediction_markets(predicted_at: str = None) -> list:
    """
    指定タイムスタンプの予測結果に含まれるマーケット一覧を返す。

    Args:
        predicted_at: Noneなら最新

    Returns:
        マーケット名のリスト
    """
    con = get_connection()

    if predicted_at is None:
        predicted_at = load_latest_prediction_timestamp()
        if predicted_at is None:
            return []

    try:
        result = con.execute(
            "SELECT DISTINCT market FROM prediction_results WHERE predicted_at = ? ORDER BY market",
            [predicted_at],
        ).fetchall()
        return [row[0] for row in result]
    except Exception as e:
        logger.error(f"予測マーケット一覧取得失敗: {e}", exc_info=True)
        return []


# --- market_data_raw テーブル操作 ---


def upsert_raw_ohlcv(rows: list[dict]) -> int:
    """
    生OHLCVデータをmarket_data_rawテーブルにUPSERT保存する。
    同一 (market, symbol, timeframe, ts) は上書きされる（べき等）。

    Args:
        rows: 各行を表す辞書のリスト。必須キー: market, symbol, ticker, timeframe, ts,
              open, high, low, close, volume。任意: adj_close, source

    Returns:
        保存した行数
    """
    if not rows:
        return 0

    con = get_connection()
    df = pd.DataFrame(rows)

    # 列の正規化
    df["ts"] = pd.to_datetime(df["ts"])
    if "adj_close" not in df.columns:
        df["adj_close"] = None
    if "source" not in df.columns:
        df["source"] = "yfinance"
    df["ingested_at"] = pd.Timestamp.utcnow()

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

    # INSERT OR REPLACE (DuckDB は INSERT OR REPLACE をサポート)
    con.register("_raw_ohlcv_temp", df)
    con.execute(
        """
        INSERT OR REPLACE INTO market_data_raw
            (market, symbol, ticker, timeframe, ts,
             open, high, low, close, volume, adj_close, source, ingested_at)
        SELECT market, symbol, ticker, timeframe, ts,
               open, high, low, close, volume, adj_close, source, ingested_at
        FROM _raw_ohlcv_temp
    """
    )
    n = len(df)
    logger.info(
        f"DB保存完了: market_data_raw [{rows[0].get('market', '')}_{rows[0].get('symbol', '')}] ({n}行)"
    )
    return n


def load_raw_ohlcv(
    market: str, symbol: str, start_date=None, end_date=None, timeframe: str = "1d"
) -> Optional[pd.DataFrame]:
    """
    market_data_rawテーブルから生OHLCVを取得する。

    Args:
        market: マーケット識別子 (例: "us", "jp")
        symbol: 銘柄シンボル (例: "AAPL", "7203")
        start_date: 開始日 (str or datetime, Noneなら全期間)
        end_date: 終了日 (str or datetime, Noneなら全期間)
        timeframe: 時間軸 (default: "1d")

    Returns:
        OHLCVのDataFrame (インデックス=ts)、データなしはNone
    """
    con = get_connection()

    query = """
        SELECT ts, open, high, low, close, volume, adj_close
        FROM market_data_raw
        WHERE market = ? AND symbol = ? AND timeframe = ?
    """
    params: list = [market, symbol, timeframe]

    if start_date is not None:
        query += " AND ts >= ?"
        params.append(pd.to_datetime(start_date))
    if end_date is not None:
        query += " AND ts <= ?"
        params.append(pd.to_datetime(end_date))

    query += " ORDER BY ts"

    try:
        df = con.execute(query, params).fetchdf()
    except Exception as e:
        logger.error(f"market_data_raw読み込み失敗 [{market}_{symbol}]: {e}", exc_info=True)
        return None

    if df.empty:
        return None

    df = df.set_index("ts")
    df.index.name = "Date"
    # カラム名を yfinance 形式（先頭大文字）に揃える
    df.columns = [c.capitalize() if c != "adj_close" else "Adj Close" for c in df.columns]
    return df
