from src.utils.db._connection import _db_connection
from src.utils.db.migration_runner import (
    _discover_migrations,
    _split_statements,
    get_applied_migrations,
    run_migrations,
)


def test_discover_migrations_finds_postgres_suffixed_files(tmp_path):
    (tmp_path / "0001_baseline_postgres.sql").write_text("SELECT 1;", encoding="utf-8")
    (tmp_path / "0001_initial.sql").write_text(
        "SELECT 1;", encoding="utf-8"
    )  # DuckDB版は無視される
    (tmp_path / "0002_add_col_postgres.sql").write_text("SELECT 1;", encoding="utf-8")

    result = _discover_migrations(str(tmp_path))

    versions = [v for v, _, _ in result]
    assert versions == ["0001", "0002"]


def test_split_statements_ignores_blank_fragments():
    sql = "SELECT 1;\n\n  ;\nSELECT 2;"
    assert _split_statements(sql) == ["SELECT 1", "SELECT 2"]


def _write_migration(tmp_path, version, name, sql):
    """テスト専用のマイグレーションSQLファイルを書き出す（*_postgres.sql命名規則）。"""
    path = tmp_path / f"{version}_{name}_postgres.sql"
    path.write_text(sql, encoding="utf-8")
    return path


def test_run_migrations_applies_pending_and_creates_table(tmp_path):
    _write_migration(
        tmp_path,
        "9101",
        "create_fixture_a",
        "CREATE TABLE _test_migration_fixture_a (id INTEGER PRIMARY KEY);",
    )

    with _db_connection() as con:
        applied_count = run_migrations(con, str(tmp_path))
        table_exists = con.execute(
            "SELECT to_regclass('_test_migration_fixture_a') IS NOT NULL"
        ).fetchone()[0]
        recorded = con.execute(
            "SELECT version FROM schema_migrations WHERE version = %s", ["9101"]
        ).fetchone()

    assert applied_count == 1
    assert table_exists is True
    assert recorded == ("9101",)


def test_run_migrations_second_run_applies_nothing(tmp_path):
    _write_migration(
        tmp_path,
        "9102",
        "create_fixture_b",
        "CREATE TABLE _test_migration_fixture_b (id INTEGER PRIMARY KEY);",
    )

    with _db_connection() as con:
        first_run = run_migrations(con, str(tmp_path))
        second_run = run_migrations(con, str(tmp_path))

    assert first_run == 1
    assert second_run == 0


def test_run_migrations_applies_in_filename_order(tmp_path):
    _write_migration(
        tmp_path,
        "9201",
        "order_first",
        "INSERT INTO _test_migration_order_log (step) VALUES ('first');",
    )
    _write_migration(
        tmp_path,
        "9202",
        "order_second",
        "INSERT INTO _test_migration_order_log (step) VALUES ('second');",
    )

    with _db_connection() as con:
        con.execute(
            "CREATE TEMP TABLE _test_migration_order_log (id SERIAL PRIMARY KEY, step VARCHAR)"
        )
        run_migrations(con, str(tmp_path))
        steps = [
            row[0]
            for row in con.execute(
                "SELECT step FROM _test_migration_order_log ORDER BY id"
            ).fetchall()
        ]

    assert steps == ["first", "second"]


def test_get_applied_migrations_returns_sorted_tuples(tmp_path):
    _write_migration(
        tmp_path,
        "9301",
        "fixture_c",
        "CREATE TABLE _test_migration_fixture_c (id INTEGER);",
    )
    _write_migration(
        tmp_path,
        "9302",
        "fixture_d",
        "CREATE TABLE _test_migration_fixture_d (id INTEGER);",
    )

    with _db_connection() as con:
        run_migrations(con, str(tmp_path))
        applied = get_applied_migrations(con)

    versions = [v for v, _, _ in applied]
    assert versions == sorted(versions)

    our_entries = {(v, d) for v, d, _ in applied if v in ("9301", "9302")}
    assert our_entries == {
        ("9301", "fixture c"),
        ("9302", "fixture d"),
    }
    for _, _, applied_at in applied:
        assert applied_at
