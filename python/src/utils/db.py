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
            market              VARCHAR NOT NULL,
            symbol              VARCHAR NOT NULL,
            predicted_at        VARCHAR NOT NULL,
            current_price       DOUBLE,
            avg_pred_price      DOUBLE,
            diff_ratio          DOUBLE,
            model_count         INTEGER,
            avg_pred_price_3d   DOUBLE,
            avg_pred_price_5d   DOUBLE,
            avg_pred_price_10d  DOUBLE,
            diff_ratio_3d       DOUBLE,
            diff_ratio_5d       DOUBLE,
            diff_ratio_10d      DOUBLE,
            confluence_score    INTEGER,
            PRIMARY KEY (market, symbol, predicted_at)
        )
    """
    )
    # 既存テーブルへのマルチホライズン列の追加（べき等）
    for col, dtype in [
        ("avg_pred_price_3d", "DOUBLE"),
        ("avg_pred_price_5d", "DOUBLE"),
        ("avg_pred_price_10d", "DOUBLE"),
        ("diff_ratio_3d", "DOUBLE"),
        ("diff_ratio_5d", "DOUBLE"),
        ("diff_ratio_10d", "DOUBLE"),
        ("confluence_score", "INTEGER"),
    ]:
        try:
            con.execute(f"ALTER TABLE prediction_results ADD COLUMN IF NOT EXISTS {col} {dtype}")
        except Exception:
            pass  # DuckDB バージョンによっては IF NOT EXISTS 未対応のため握りつぶす
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
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS model_metrics (
            market               VARCHAR NOT NULL,
            symbol               VARCHAR NOT NULL,
            model_name           VARCHAR NOT NULL,
            trained_at           VARCHAR NOT NULL,
            rmse                 DOUBLE,
            directional_accuracy DOUBLE,
            n_samples            INTEGER,
            PRIMARY KEY (market, symbol, model_name, trained_at)
        )
    """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_accuracy (
            market          VARCHAR NOT NULL,
            symbol          VARCHAR NOT NULL,
            model_name      VARCHAR NOT NULL,
            predicted_at    VARCHAR NOT NULL,
            horizon         INTEGER NOT NULL DEFAULT 1,
            predicted_price DOUBLE,
            actual_price    DOUBLE,
            predicted_ratio DOUBLE,
            actual_ratio    DOUBLE,
            direction_match BOOLEAN,
            checked_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (market, symbol, model_name, predicted_at, horizon)
        )
    """
    )

    # --- 自動売買テーブル ---
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS orders (
            order_id    VARCHAR NOT NULL,
            symbol      VARCHAR NOT NULL,
            side        INTEGER NOT NULL,
            qty         INTEGER NOT NULL,
            price       DOUBLE NOT NULL DEFAULT 0.0,
            order_type  INTEGER NOT NULL DEFAULT 10,
            fill_price  DOUBLE,
            status      VARCHAR NOT NULL DEFAULT 'pending',
            broker      VARCHAR NOT NULL,
            mode        VARCHAR NOT NULL DEFAULT 'paper',
            created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            filled_at   TIMESTAMP,
            PRIMARY KEY (order_id)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_orders (
            order_id    VARCHAR NOT NULL,
            symbol      VARCHAR NOT NULL,
            side        INTEGER NOT NULL,
            qty         INTEGER NOT NULL,
            price       DOUBLE NOT NULL DEFAULT 0.0,
            order_type  INTEGER NOT NULL DEFAULT 10,
            fill_price  DOUBLE,
            status      VARCHAR NOT NULL DEFAULT 'pending',
            created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            filled_at   TIMESTAMP,
            PRIMARY KEY (order_id)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_positions (
            symbol      VARCHAR NOT NULL PRIMARY KEY,
            qty         INTEGER NOT NULL DEFAULT 0,
            avg_price   DOUBLE NOT NULL DEFAULT 0.0,
            updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_balance (
            id          INTEGER NOT NULL PRIMARY KEY DEFAULT 1,
            balance     DOUBLE NOT NULL DEFAULT 1000000.0
        )
        """
    )
    # 初期残高レコードが存在しない場合のみ挿入
    if con.execute("SELECT COUNT(*) FROM paper_balance").fetchone()[0] == 0:
        con.execute("INSERT INTO paper_balance (id, balance) VALUES (1, 1000000.0)")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_pnl (
            trade_id     VARCHAR NOT NULL PRIMARY KEY,
            symbol       VARCHAR NOT NULL,
            side         INTEGER NOT NULL,
            qty          INTEGER NOT NULL,
            entry_price  DOUBLE NOT NULL,
            exit_price   DOUBLE NOT NULL,
            realized_pnl DOUBLE NOT NULL,
            broker       VARCHAR NOT NULL,
            mode         VARCHAR NOT NULL DEFAULT 'paper',
            opened_at    TIMESTAMP,
            closed_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
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

    # 必須列
    base_cols = [
        "market",
        "symbol",
        "predicted_at",
        "current_price",
        "avg_pred_price",
        "diff_ratio",
        "model_count",
    ]
    # マルチホライズン列（存在する場合のみ）
    multi_horizon_cols = [
        "avg_pred_price_3d",
        "avg_pred_price_5d",
        "avg_pred_price_10d",
        "diff_ratio_3d",
        "diff_ratio_5d",
        "diff_ratio_10d",
        "confluence_score",
    ]
    extra_cols = [c for c in multi_horizon_cols if c in save_df.columns]
    cols = base_cols + extra_cols

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

    # 列名を明示して INSERT（テーブル列数とのミスマッチ回避）
    col_str = ", ".join(cols)
    con.register("_save_df_temp", save_df)
    con.execute(f"INSERT INTO prediction_results ({col_str}) SELECT {col_str} FROM _save_df_temp")
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


# --- model_metrics テーブル操作 ---


def save_model_metrics(
    market: str,
    symbol: str,
    model_name: str,
    trained_at: str,
    metrics: dict,
) -> None:
    """
    モデル学習後の精度指標を model_metrics テーブルに保存する。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
        model_name: モデル名 (ex: "StockXGBoostModel")
        trained_at: 学習日時文字列 (ex: "20260314_120000")
        metrics: {"rmse": float, "directional_accuracy": float, "n_samples": int}
    """
    con = get_connection()
    con.execute(
        """
        INSERT OR REPLACE INTO model_metrics
            (market, symbol, model_name, trained_at, rmse, directional_accuracy, n_samples)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            market,
            symbol,
            model_name,
            trained_at,
            metrics.get("rmse"),
            metrics.get("directional_accuracy"),
            metrics.get("n_samples"),
        ],
    )
    logger.debug(
        f"model_metrics保存: [{market}_{symbol}/{model_name}] "
        f"RMSE={metrics.get('rmse', 'N/A'):.6f}, "
        f"方向正解率={metrics.get('directional_accuracy', 'N/A'):.2%}"
    )


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


def load_all_raw_ohlcv_symbols(timeframe: str = "1d") -> list:
    """
    market_data_raw テーブルに存在する全 (market, symbol) のリストを返す。

    Args:
        timeframe: 時間軸 (default: "1d")

    Returns:
        [(market, symbol), ...] のリスト
    """
    con = get_connection()
    try:
        sql = (
            "SELECT DISTINCT market, symbol FROM market_data_raw"
            " WHERE timeframe = ? ORDER BY market, symbol"
        )
        rows = con.execute(sql, [timeframe]).fetchall()
        return [(r[0], r[1]) for r in rows]
    except Exception as e:
        logger.error(f"load_all_raw_ohlcv_symbols 失敗: {e}", exc_info=True)
        return []


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


# --- prediction_accuracy テーブル操作 ---


def save_prediction_accuracy(rows: list[dict]) -> int:
    """
    予測精度データを prediction_accuracy テーブルへ保存（UPSERT）する。

    Args:
        rows: 各行は以下のキーを持つ dict
            market, symbol, model_name, predicted_at, horizon,
            predicted_price, actual_price, predicted_ratio, actual_ratio, direction_match

    Returns:
        挿入/更新件数
    """
    if not rows:
        return 0

    con = get_connection()
    inserted = 0
    for row in rows:
        try:
            con.execute(
                """
                INSERT OR REPLACE INTO prediction_accuracy
                    (market, symbol, model_name, predicted_at, horizon,
                     predicted_price, actual_price, predicted_ratio, actual_ratio,
                     direction_match, checked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [
                    row["market"],
                    row["symbol"],
                    row["model_name"],
                    row["predicted_at"],
                    row.get("horizon", 1),
                    row.get("predicted_price"),
                    row.get("actual_price"),
                    row.get("predicted_ratio"),
                    row.get("actual_ratio"),
                    row.get("direction_match"),
                ],
            )
            inserted += 1
        except Exception as e:
            logger.error(
                f"prediction_accuracy 保存失敗 [{row.get('market')}_{row.get('symbol')}]: {e}",
                exc_info=True,
            )

    logger.info(f"prediction_accuracy 保存完了: {inserted}件")
    return inserted


def load_prediction_accuracy(
    market: str = None,
    symbol: str = None,
    horizon: int = 1,
    limit: int = 500,
) -> pd.DataFrame:
    """
    prediction_accuracy テーブルからデータを取得する。

    Args:
        market: フィルタ対象の市場名（None なら全市場）
        symbol: フィルタ対象の銘柄コード（None なら全銘柄）
        horizon: フィルタ対象のホライズン（デフォルト 1）
        limit: 返却行数の上限

    Returns:
        pd.DataFrame（結果なし時は空 DataFrame）
    """
    con = get_connection()
    query = "SELECT * FROM prediction_accuracy WHERE horizon = ?"
    params: list = [horizon]
    if market is not None:
        query += " AND market = ?"
        params.append(market)
    if symbol is not None:
        query += " AND symbol = ?"
        params.append(symbol)
    query += " ORDER BY predicted_at DESC"
    if limit:
        query += f" LIMIT {int(limit)}"
    try:
        return con.execute(query, params).fetchdf()
    except Exception as e:
        logger.error(f"prediction_accuracy 読み込み失敗: {e}", exc_info=True)
        return pd.DataFrame()


def load_drift_summary(horizon: int = 1, recent_n: int = 30) -> pd.DataFrame:
    """
    直近 recent_n 件の予測について銘柄ごとの方向正解率・平均誤差を集計する。

    Args:
        horizon: 対象ホライズン
        recent_n: 直近何件を対象にするか（銘柄ごとに最新 N 件）

    Returns:
        pd.DataFrame: [market, symbol, direction_accuracy, mean_abs_error, n_samples]
    """
    con = get_connection()
    try:
        df = con.execute(
            f"""
            WITH ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY market, symbol ORDER BY predicted_at DESC) AS rn
                FROM prediction_accuracy
                WHERE horizon = ?
            )
            SELECT
                market,
                symbol,
                AVG(CAST(direction_match AS INTEGER)) AS direction_accuracy,
                AVG(ABS(predicted_ratio - actual_ratio))  AS mean_abs_error,
                COUNT(*) AS n_samples
            FROM ranked
            WHERE rn <= {int(recent_n)}
            GROUP BY market, symbol
            ORDER BY direction_accuracy ASC
            """,
            [horizon],
        ).fetchdf()
        return df
    except Exception as e:
        logger.error(f"load_drift_summary 失敗: {e}", exc_info=True)
        return pd.DataFrame()
