"""
DuckDB 接続管理・スキーマ初期化モジュール

短命接続パターンを採用し、DBファイルのロック時間を最小化する。
各DB操作は _db_connection() コンテキストマネージャーを通じて接続を取得し、
操作完了後すぐに接続を閉じる。

Docker 上の複数プロセス（スケジューラー・API サーバー等）が並行動作する環境での
ロック衝突を防ぐために採用した設計。
"""

import time
from contextlib import contextmanager
from threading import Lock
from typing import Generator

import duckdb
from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from src.utils.data_path_utils import ensure_dir, get_data_dir, get_db_path
from src.utils.db.migration_runner import run_migrations
from src.utils.logger import get_logger

logger = get_logger(__name__)

# --- 設定 ---
_RETRY_COUNT = 10  # ロック衝突時の最大リトライ回数
_RETRY_DELAY = 1.0  # リトライ間隔（秒）
_DB_CONFIG = {"threads": "4", "memory_limit": "2GB"}
_FILELOCK_TIMEOUT = 120.0  # プロセス間mutex タイムアウト（秒）

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
    lock_path = db_path + ".lock"

    file_lock = FileLock(lock_path, timeout=_FILELOCK_TIMEOUT)
    try:
        file_lock.acquire()
    except FileLockTimeout:
        raise RuntimeError(
            f"DuckDB書き込みロック取得タイムアウト ({_FILELOCK_TIMEOUT}秒): 別プロセスがDBを使用中です。{lock_path}"
        )

    con = None
    try:
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
        # テーブル初期化: プロセス起動後の初回接続時のみ実行（ダブルチェックロッキング）
        if not _tables_initialized:
            with _init_lock:
                if not _tables_initialized:
                    _init_tables(con)
                    run_migrations(con)
                    _tables_initialized = True
        yield con
    finally:
        if con is not None:
            con.close()
        file_lock.release()


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
            market              VARCHAR NOT NULL,
            symbol              VARCHAR NOT NULL,
            predicted_at        VARCHAR NOT NULL,
            model_version       VARCHAR NOT NULL DEFAULT 'production',
            run_id              VARCHAR,
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
            PRIMARY KEY (market, symbol, predicted_at, model_version)
        )
    """)
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
        ("model_version", "VARCHAR"),
        ("run_id", "VARCHAR"),
    ]:
        try:
            con.execute(f"ALTER TABLE prediction_results ADD COLUMN IF NOT EXISTS {col} {dtype}")
        except Exception:
            logger.debug("ALTER TABLE スキップ（DuckDB互換）: col=%s", col, exc_info=True)
    con.execute("""
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
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS index_membership_history (
            market        VARCHAR NOT NULL,
            symbol        VARCHAR NOT NULL,
            index_name    VARCHAR NOT NULL,
            snapshot_date DATE NOT NULL,
            source        VARCHAR NOT NULL DEFAULT 'wikipedia',
            fetched_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (market, symbol, snapshot_date)
        )
    """)
    con.execute("""
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
    """)
    con.execute("""
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
    """)
    # ペーパートレード用テーブル
    con.execute("""
        CREATE TABLE IF NOT EXISTS paper_balance (
            balance DOUBLE NOT NULL
        )
    """)
    # paper_balanceが空なら初期残高（100万円）を挿入
    count = con.execute("SELECT COUNT(*) FROM paper_balance").fetchone()[0]
    if count == 0:
        con.execute("INSERT INTO paper_balance VALUES (1000000.0)")

    con.execute("""
        CREATE TABLE IF NOT EXISTS paper_orders (
            order_id     VARCHAR NOT NULL PRIMARY KEY,
            market       VARCHAR,
            predicted_at VARCHAR,
            symbol       VARCHAR NOT NULL,
            side         INTEGER NOT NULL,
            qty          INTEGER NOT NULL,
            price        DOUBLE,
            signal_price DOUBLE,
            order_type   INTEGER NOT NULL,
            status       VARCHAR NOT NULL DEFAULT 'pending',
            fill_price   DOUBLE,
            realized_pnl DOUBLE,
            filled_at    TIMESTAMP,
            created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # realized_pnl カラムが既存テーブルに存在しない場合にマイグレーション
    existing_cols = [
        row[0]
        for row in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='paper_orders'"
        ).fetchall()
    ]
    if "realized_pnl" not in existing_cols:
        con.execute("ALTER TABLE paper_orders ADD COLUMN realized_pnl DOUBLE")
    for col, dtype in [
        ("market", "VARCHAR"),
        ("predicted_at", "VARCHAR"),
        ("signal_price", "DOUBLE"),
    ]:
        if col not in existing_cols:
            con.execute(f"ALTER TABLE paper_orders ADD COLUMN {col} {dtype}")
    con.execute("""
        CREATE TABLE IF NOT EXISTS paper_positions (
            symbol      VARCHAR NOT NULL PRIMARY KEY,
            qty         INTEGER NOT NULL,
            avg_price   DOUBLE NOT NULL,
            updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.execute("""
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
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS paper_real_diff (
            market          VARCHAR NOT NULL,
            symbol          VARCHAR NOT NULL,
            predicted_at    VARCHAR NOT NULL,
            side            INTEGER NOT NULL,
            signal_price    DOUBLE,
            paper_order_id  VARCHAR,
            real_order_id   VARCHAR,
            paper_price     DOUBLE,
            real_price      DOUBLE,
            paper_slippage  DOUBLE,
            real_slippage   DOUBLE,
            price_diff      DOUBLE,
            paper_filled_at TIMESTAMP,
            real_checked_at TIMESTAMP,
            created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            order_session   VARCHAR,
            split_ratio     DOUBLE,
            PRIMARY KEY (market, symbol, predicted_at, side)
        )
    """)
    # R-405: order_session カラムのマイグレーション（既存テーブル対応）
    prd_cols = [
        row[0]
        for row in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='paper_real_diff'"
        ).fetchall()
    ]
    if "order_session" not in prd_cols:
        con.execute("ALTER TABLE paper_real_diff ADD COLUMN order_session VARCHAR")
    if "split_ratio" not in prd_cols:
        con.execute("ALTER TABLE paper_real_diff ADD COLUMN split_ratio DOUBLE")
    con.execute("""
        CREATE TABLE IF NOT EXISTS feature_selection_log (
            market             VARCHAR NOT NULL,
            symbol             VARCHAR NOT NULL,
            model_name         VARCHAR NOT NULL,
            trained_at         VARCHAR NOT NULL,
            feature            VARCHAR NOT NULL,
            importance_mean    DOUBLE NOT NULL,
            importance_std     DOUBLE NOT NULL,
            importance_rank    INTEGER NOT NULL,
            is_excluded        BOOLEAN NOT NULL DEFAULT FALSE,
            protected_by_shap  BOOLEAN NOT NULL DEFAULT FALSE,
            PRIMARY KEY (market, symbol, model_name, trained_at, feature)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS experiment_runs (
            run_id               VARCHAR NOT NULL PRIMARY KEY,
            market               VARCHAR NOT NULL,
            symbol               VARCHAR NOT NULL,
            model_name           VARCHAR NOT NULL,
            trained_at           VARCHAR NOT NULL,
            horizon              INTEGER NOT NULL DEFAULT 1,
            rmse                 DOUBLE,
            directional_accuracy DOUBLE,
            n_samples            INTEGER,
            n_features           INTEGER,
            feature_hash         VARCHAR,
            params_json          VARCHAR,
            created_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # R-214: 発注実行サマリーテーブル
    con.execute("""
        CREATE TABLE IF NOT EXISTS order_run_summary (
            run_id            VARCHAR   NOT NULL PRIMARY KEY,
            market            VARCHAR   NOT NULL,
            mode              VARCHAR   NOT NULL,
            run_at            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            buy_orders        INTEGER   NOT NULL DEFAULT 0,
            sell_orders       INTEGER   NOT NULL DEFAULT 0,
            short_orders      INTEGER   NOT NULL DEFAULT 0,
            skipped           INTEGER   NOT NULL DEFAULT 0,
            skipped_min_change INTEGER  NOT NULL DEFAULT 0,
            total_turnover    DOUBLE    NOT NULL DEFAULT 0.0,
            min_change_ratio  DOUBLE
        )
    """)

    # R-215: 空売りポジションテーブル
    con.execute("""
        CREATE TABLE IF NOT EXISTS paper_short_positions (
            symbol            VARCHAR   NOT NULL PRIMARY KEY,
            qty               INTEGER   NOT NULL,
            avg_short_price   DOUBLE    NOT NULL,
            unrealized_pnl    DOUBLE,
            opened_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS dd_state (
            id            INTEGER PRIMARY KEY,
            peak_balance  DOUBLE  NOT NULL
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS data_quality_log (
            market      VARCHAR NOT NULL,
            symbol      VARCHAR NOT NULL,
            check_name  VARCHAR NOT NULL,
            level       VARCHAR NOT NULL,
            detail      VARCHAR NOT NULL,
            checked_at  VARCHAR NOT NULL
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS system_config (
            key        VARCHAR NOT NULL PRIMARY KEY,
            value      VARCHAR NOT NULL,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # I-241: 週次精度スナップショット
    con.execute("""
        CREATE TABLE IF NOT EXISTS accuracy_weekly_snapshots (
            week_start         VARCHAR NOT NULL,
            market             VARCHAR NOT NULL,
            symbol             VARCHAR NOT NULL,
            direction_accuracy DOUBLE,
            mean_abs_error     DOUBLE,
            n_samples          INTEGER,
            snapshot_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (week_start, market, symbol)
        )
    """)


def init_tables() -> None:
    """外部から明示的にテーブル初期化する場合に使用"""
    with _db_connection() as con:
        _init_tables(con)
