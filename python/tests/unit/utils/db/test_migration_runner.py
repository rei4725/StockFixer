from src.utils.db.migration_runner import _discover_migrations, _split_statements


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
