"""
DuckDB 接続管理・スキーマ初期化モジュール

短命接続パターンを採用し、DBファイルのロック時間を最小化する。
各DB操作は _db_connection() コンテキストマネージャーを通じて接続を取得し、
操作完了後すぐに接続を閉じる。

Docker 上の複数プロセス（スケジューラー・API サーバー等）が並行動作する環境での
ロック衝突を防ぐために採用した設計。
"""

import time
import warnings
from contextlib import contextmanager
from threading import Lock
from typing import Generator

import duckdb

from src.utils.data_path_utils import ensure_dir, get_data_dir, get_db_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

# --- 設定 ---
_RETRY_COUNT = 10  # ロック衝突時の最大リトライ回数
_RETRY_DELAY = 1.0  # リトライ間隔（秒）
_DB_CONFIG = {"threads": "4", "memory_limit": "2GB"}

_init_lock = Lock()  # テーブル初期化の二重実行防止
_tables_initialized = False  # プロセス内でのテーブル初期化済みフラグ


@contextmanager
def _db_connection() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """
    短命DB接続を提供するコンテキストマネージャー。

    with ブロックの間だけ接続を保持し、終了後すぐ閉じる。
    DuckDB は読み書き接続をプロセスレベルで 1 つしか持てないため、
    接続を長時間保持すると他のプロセスからアクセスできなくなる。

    リトライ: IOException（ロック衝突等）が発生した場合は
    _RETRY_COUNT 回まで _RETRY_DELAY 秒待ってリトライする。

    Usage:
        with _db_connection() as con:
            df = con.execute("SELECT ...").fetchdf()
    """
    global _tables_initialized

    ensure_dir(get_data_dir())
    db_path = get_db_path()

    con = None
    last_exc: Exception = RuntimeError("DB接続に失敗しました")
    for attempt in range(_RETRY_COUNT):
        try:
            con = duckdb.connect(db_path, config=_DB_CONFIG)
            break
        except (duckdb.IOException, duckdb.BinderException) as e:
            last_exc = e
            if attempt < _RETRY_COUNT - 1:
                logger.warning(f"DB接続待機中 ({attempt + 1}/{_RETRY_COUNT}): {e}")
                time.sleep(_RETRY_DELAY)

    if con is None:
        raise last_exc

    try:
        # テーブル初期化: プロセス起動後の初回接続時のみ実行（ダブルチェックロッキング）
        if not _tables_initialized:
            with _init_lock:
                if not _tables_initialized:
                    _init_tables(con)
                    _tables_initialized = True
        yield con
    finally:
        con.close()


def get_connection() -> duckdb.DuckDBPyConnection:
    """
    後方互換用: 新規の読み書き接続を返す。呼び出し側で close() すること。

    .. deprecated::
        `_db_connection()` コンテキストマネージャーを推奨。
        接続の閉じ忘れによるロック残留を防ぐため with 文を使用してください。
    """
    warnings.warn(
        "get_connection() は非推奨です。_db_connection() コンテキストマネージャーを使用してください。",
        DeprecationWarning,
        stacklevel=2,
    )
    ensure_dir(get_data_dir())
    db_path = get_db_path()
    con = duckdb.connect(db_path, config=_DB_CONFIG)
    if not _tables_initialized:
        _init_tables(con)
    return con


def close_connection() -> None:
    """
    状態リセット。短命接続モデルでは永続接続は持たないが、
    _tables_initialized フラグをリセットしてテスト間のDB切り替えに対応する。
    """
    global _tables_initialized
    _tables_initialized = False


def get_readonly_connection() -> duckdb.DuckDBPyConnection:
    """
    読み取り専用の新規接続を返す（別プロセスからの利用向け）。
    呼び出し側で close() すること。
    """
    ensure_dir(get_data_dir())
    db_path = get_db_path()
    return duckdb.connect(db_path, read_only=True)


# ---------------------------------------------------------------------------
# テーブル DDL
# ---------------------------------------------------------------------------


def _init_tables(con: duckdb.DuckDBPyConnection) -> None:
    """全テーブルを作成する（CREATE TABLE IF NOT EXISTS）"""
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
            confidence_ratio    DOUBLE,
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
    # 既存テーブルへのマルチホライズン列追加（べき等）
    for col, dtype in [
        ("avg_pred_price_3d", "DOUBLE"),
        ("avg_pred_price_5d", "DOUBLE"),
        ("avg_pred_price_10d", "DOUBLE"),
        ("diff_ratio_3d", "DOUBLE"),
        ("diff_ratio_5d", "DOUBLE"),
        ("diff_ratio_10d", "DOUBLE"),
        ("confluence_score", "INTEGER"),
        ("confidence_ratio", "DOUBLE"),
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
    # ペーパートレード用テーブル
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_balance (
            balance DOUBLE NOT NULL
        )
    """
    )
    # paper_balanceが空なら初期残高（100万円）を挿入
    count = con.execute("SELECT COUNT(*) FROM paper_balance").fetchone()[0]
    if count == 0:
        con.execute("INSERT INTO paper_balance VALUES (1000000.0)")

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_orders (
            order_id     VARCHAR NOT NULL PRIMARY KEY,
            symbol       VARCHAR NOT NULL,
            side         INTEGER NOT NULL,
            qty          INTEGER NOT NULL,
            price        DOUBLE,
            order_type   INTEGER NOT NULL,
            status       VARCHAR NOT NULL DEFAULT 'pending',
            fill_price   DOUBLE,
            realized_pnl DOUBLE,
            filled_at    TIMESTAMP,
            created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    # realized_pnl カラムが既存テーブルに存在しない場合にマイグレーション
    existing_cols = [
        row[0]
        for row in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='paper_orders'"
        ).fetchall()
    ]
    if "realized_pnl" not in existing_cols:
        con.execute("ALTER TABLE paper_orders ADD COLUMN realized_pnl DOUBLE")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_positions (
            symbol      VARCHAR NOT NULL PRIMARY KEY,
            qty         INTEGER NOT NULL,
            avg_price   DOUBLE NOT NULL,
            updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS shap_values (
            market      VARCHAR NOT NULL,
            symbol      VARCHAR NOT NULL,
            model_name  VARCHAR NOT NULL,
            trained_at  VARCHAR NOT NULL,
            feature     VARCHAR NOT NULL,
            shap_mean   DOUBLE NOT NULL,
            shap_rank   INTEGER NOT NULL,
            PRIMARY KEY (market, symbol, model_name, trained_at, feature)
        )
    """
    )


def init_tables() -> None:
    """外部から明示的にテーブル初期化する場合に使用"""
    with _db_connection() as con:
        _init_tables(con)
