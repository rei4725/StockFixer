"""verify_postgres_migration.py の統合テスト（実DuckDB・実Postgres両方を使用）。

重要: `verify_table` はDuckDBの `ATTACH ... (TYPE POSTGRES)` 経由でPostgresへ直接
問い合わせる。これは `_isolate_db`（tests/integration/conftest.py）が
`set_test_connection` で差し替える psycopg 接続とは別物の libpq 接続であり、
`_isolate_db` が注入する接続へ書いた行（未コミット）はこの別接続からは見えない
（標準的な READ COMMITTED の可視性ルール）。そのため、Postgres側の比較対象データは
`_db_connection()` 経由ではなく、直接コミットする psycopg 接続で用意し、テスト側の
`finally` で明示的に後始末する（`test_migrate_to_postgres.py` と同じパターン）。
"""

import duckdb
import psycopg
import pytest
from scripts.verify_postgres_migration import verify_table

from src.utils.data_path_utils import get_database_url


def _attach(db_path: str) -> duckdb.DuckDBPyConnection:
    src_con = duckdb.connect(db_path, read_only=True)
    src_con.execute("INSTALL postgres")
    src_con.execute("LOAD postgres")
    src_con.execute(f"ATTACH '{get_database_url()}' AS pg (TYPE POSTGRES)")
    return src_con


# ============================================
# COUNT(*) 照合（SUM対象外テーブル: dd_state）
# ============================================


@pytest.fixture
def dd_state_duckdb(tmp_path):
    db_path = str(tmp_path / "sample.duckdb")
    con = duckdb.connect(db_path)
    con.execute("CREATE TABLE dd_state (id INTEGER PRIMARY KEY, peak_balance DOUBLE)")
    con.execute("INSERT INTO dd_state VALUES (1, 500.0)")
    con.close()
    return db_path


@pytest.fixture
def dd_state_pg_row():
    """Postgres側の dd_state に1行を実コミットで投入し、テスト後にTRUNCATEする。

    `_isolate_db` の注入接続はロールバックされるだけで別接続からは見えないため、
    DuckDB の ATTACH 経由で読める状態を作るには直接コミットする必要がある。
    """
    with psycopg.connect(get_database_url(), autocommit=True) as con:
        con.execute("TRUNCATE TABLE dd_state")
        con.execute("INSERT INTO dd_state (id, peak_balance) VALUES (1, 500.0)")
    try:
        yield
    finally:
        with psycopg.connect(get_database_url(), autocommit=True) as con:
            con.execute("TRUNCATE TABLE dd_state")


def test_verify_table_passes_when_counts_match(dd_state_duckdb, dd_state_pg_row, _isolate_db):
    src_con = _attach(dd_state_duckdb)
    try:
        assert verify_table(src_con, "dd_state") is True
    finally:
        src_con.close()


def test_verify_table_fails_when_counts_differ(dd_state_duckdb, _isolate_db):
    """Postgres側に対応行を投入しないため、DuckDB=1行 / Postgres=0行で不一致になる。"""
    with psycopg.connect(get_database_url(), autocommit=True) as con:
        con.execute("TRUNCATE TABLE dd_state")

    src_con = _attach(dd_state_duckdb)
    try:
        assert verify_table(src_con, "dd_state") is False
    finally:
        src_con.close()


# ============================================
# SUM(...) 照合（_SUM_CHECKS 対象の3テーブル）
# ============================================


@pytest.fixture
def paper_balance_pg_snapshot():
    """paper_balance はベースライン移行で唯一行(balance=1000000.0)が
    シードされる想定のテーブル。既存の値を読むだけで、書き換えは行わない
    （他の統合テストが前提にしている可能性のある状態を壊さないため）。
    """
    with psycopg.connect(get_database_url(), autocommit=True) as con:
        rows = con.execute("SELECT balance FROM paper_balance").fetchall()
    assert (
        len(rows) == 1
    ), f"想定外の paper_balance 行数: {len(rows)}（シード1行のみを前提とするテスト）"
    return rows[0][0]


def test_verify_table_passes_sum_check_for_paper_balance(
    paper_balance_pg_snapshot, tmp_path, _isolate_db
):
    """SUM照合の一致（OK）経路を paper_balance で確認する。Postgres側は既存のシード行
    をそのまま使い、DuckDB側にそれと同額の1行を用意して一致させる。
    """
    db_path = str(tmp_path / "sample_paper_balance.duckdb")
    con = duckdb.connect(db_path)
    con.execute("CREATE TABLE paper_balance (balance DOUBLE)")
    con.execute("INSERT INTO paper_balance VALUES (?)", [paper_balance_pg_snapshot])
    con.close()

    src_con = _attach(db_path)
    try:
        assert verify_table(src_con, "paper_balance") is True
    finally:
        src_con.close()


@pytest.fixture
def paper_positions_pg_row():
    """Postgres側の paper_positions に qty=100 の1行を実コミットで投入する。"""
    with psycopg.connect(get_database_url(), autocommit=True) as con:
        con.execute("TRUNCATE TABLE paper_positions")
        con.execute(
            "INSERT INTO paper_positions (symbol, qty, avg_price, updated_at) "
            "VALUES ('AAPL', 100, 150.0, CURRENT_TIMESTAMP)"
        )
    try:
        yield
    finally:
        with psycopg.connect(get_database_url(), autocommit=True) as con:
            con.execute("TRUNCATE TABLE paper_positions")


def test_verify_table_fails_sum_check_for_paper_positions_qty(
    paper_positions_pg_row, tmp_path, _isolate_db
):
    """件数(COUNT)は一致するが SUM(qty) が不一致になるケース。
    件数照合をすり抜けて `_SUM_CHECKS` ループが実際に不一致を検出することを確認する。
    """
    db_path = str(tmp_path / "sample_paper_positions.duckdb")
    con = duckdb.connect(db_path)
    con.execute(
        "CREATE TABLE paper_positions "
        "(symbol VARCHAR, qty INTEGER, avg_price DOUBLE, updated_at TIMESTAMP)"
    )
    # Postgres側は qty=100（同一symbol 1行）。DuckDB側はあえて qty=999 とし、
    # 行数(1行 vs 1行)は一致させたまま合計値だけを不一致にする。
    con.execute("INSERT INTO paper_positions VALUES ('AAPL', 999, 150.0, CURRENT_TIMESTAMP)")
    con.close()

    src_con = _attach(db_path)
    try:
        assert verify_table(src_con, "paper_positions") is False
    finally:
        src_con.close()


@pytest.fixture
def paper_orders_pg_row():
    """Postgres側の paper_orders に realized_pnl=1000.0 の1行を実コミットで投入する。"""
    with psycopg.connect(get_database_url(), autocommit=True) as con:
        con.execute("TRUNCATE TABLE paper_orders")
        con.execute(
            "INSERT INTO paper_orders "
            "(order_id, symbol, side, qty, order_type, status, realized_pnl) "
            "VALUES ('ord-1', 'AAPL', 0, 10, 0, 'filled', 1000.0)"
        )
    try:
        yield
    finally:
        with psycopg.connect(get_database_url(), autocommit=True) as con:
            con.execute("TRUNCATE TABLE paper_orders")


def test_verify_table_fails_sum_check_for_paper_orders_realized_pnl(
    paper_orders_pg_row, tmp_path, _isolate_db
):
    """件数(COUNT)は一致するが SUM(realized_pnl) が不一致になるケース。"""
    db_path = str(tmp_path / "sample_paper_orders.duckdb")
    con = duckdb.connect(db_path)
    con.execute(
        "CREATE TABLE paper_orders "
        "(order_id VARCHAR, symbol VARCHAR, side INTEGER, qty INTEGER, "
        "order_type INTEGER, status VARCHAR, realized_pnl DOUBLE)"
    )
    # Postgres側は realized_pnl=1000.0（同一order_id 1行）。DuckDB側は 1件は揃えつつ
    # 合計値だけを不一致にする。
    con.execute("INSERT INTO paper_orders VALUES ('ord-1', 'AAPL', 0, 10, 0, 'filled', 250.0)")
    con.close()

    src_con = _attach(db_path)
    try:
        assert verify_table(src_con, "paper_orders") is False
    finally:
        src_con.close()
