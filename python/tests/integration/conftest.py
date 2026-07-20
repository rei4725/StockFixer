# Integration test専用 fixture。実DB・外部ライブラリ依存を含む。
import pytest

# ============================================
# 環境チェック Marker
# ============================================


@pytest.fixture(scope="session")
def has_xgboost():
    """Check if XGBoost is available."""
    try:
        import xgboost  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def has_duckdb():
    """Check if DuckDB is available."""
    try:
        import duckdb  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def _test_database_ready():
    from src.utils.db._connection import _get_pool
    from src.utils.db.migration_runner import run_migrations

    with _get_pool().connection() as con:
        run_migrations(con)
    yield


@pytest.fixture(autouse=True)
def _isolate_db(_test_database_ready):
    import psycopg

    from src.utils.data_path_utils import get_database_url
    from src.utils.db._connection import close_connection, set_test_connection

    con = psycopg.connect(get_database_url(), autocommit=False)
    set_test_connection(con)
    try:
        yield
    finally:
        con.rollback()
        set_test_connection(None)
        con.close()
        close_connection()
