"""
PostgreSQL 接続管理モジュール（psycopg3 + コネクションプール）

通常運用ではプロセス単位のコネクションプールから接続を借用する。
DuckDB版が暗黙のautocommitで動いていた（呼び出し側は一切 commit() しない）
挙動をそのまま踏襲するため、プール接続は autocommit=True で払い出す。

テスト時は set_test_connection() で単一の共有接続（autocommit=False）を
注入できる。呼び出し側が commit() しない設計のため、テスト終了時に
その接続を rollback() するだけで全ての変更を巻き戻せる。
"""

from contextlib import contextmanager
from typing import Generator, Optional

import psycopg
from psycopg_pool import ConnectionPool, PoolTimeout

from src.utils.data_path_utils import get_database_url
from src.utils.db.migration_runner import run_migrations
from src.utils.logger import get_logger

logger = get_logger(__name__)

_pool: Optional[ConnectionPool] = None
_tables_initialized = False
_test_connection: Optional[psycopg.Connection] = None

_DEFAULT_LOCK_TIMEOUT = 30.0

# 初回スキーマ・マイグレーション適用をプロセス間で直列化するための固定キー。
# 任意の bigint 定数（このコードベースの他箇所では未使用）。
_MIGRATION_LOCK_KEY = 727100


class DbLockTimeoutError(RuntimeError):
    """コネクションプールからの接続取得がタイムアウトしたことを表す。

    「別処理がDBを使用中で空き接続がない」ことを意味し、DB自体の異常ではない。
    ヘルスチェック等が busy と異常を区別するために使う。
    """


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            get_database_url(),
            min_size=1,
            max_size=4,
            kwargs={"autocommit": True},
            open=True,
        )
    return _pool


@contextmanager
def _db_connection(
    lock_timeout: float | None = None,
) -> Generator[psycopg.Connection, None, None]:
    """
    DB接続を提供するコンテキストマネージャー。

    Usage:
        with _db_connection() as con:
            df = con.execute("SELECT ...").fetchall()
    """
    global _tables_initialized

    if _test_connection is not None:
        yield _test_connection
        return

    timeout = _DEFAULT_LOCK_TIMEOUT if lock_timeout is None else lock_timeout
    try:
        with _get_pool().connection(timeout=timeout) as con:
            if not _tables_initialized:
                con.execute("SELECT pg_advisory_lock(%s)", [_MIGRATION_LOCK_KEY])
                try:
                    run_migrations(con)
                    _tables_initialized = True
                finally:
                    con.execute("SELECT pg_advisory_unlock(%s)", [_MIGRATION_LOCK_KEY])
            yield con
    except PoolTimeout as e:
        raise DbLockTimeoutError(
            f"PostgreSQLコネクションプールからの接続取得がタイムアウトしました ({timeout}秒): {e}"
        ) from e


def set_test_connection(con: Optional[psycopg.Connection]) -> None:
    """テスト専用フック。以降の _db_connection() 呼び出しをこの接続に固定する。"""
    global _test_connection
    _test_connection = con


def close_connection() -> None:
    """状態リセット。プールを閉じ、テーブル初期化フラグを戻す（テスト間の切り替え用）"""
    global _tables_initialized, _pool
    _tables_initialized = False
    if _pool is not None:
        _pool.close()
        _pool = None


def get_readonly_connection() -> psycopg.Connection:
    """
    読み取り専用の新規接続を返す（呼び出し側で close() すること）。
    プールを介さない単発接続。
    """
    con = psycopg.connect(get_database_url(), autocommit=True)
    con.read_only = True
    return con


def init_tables() -> None:
    """外部から明示的にテーブル初期化する場合に使用"""
    with _db_connection() as con:
        run_migrations(con)
