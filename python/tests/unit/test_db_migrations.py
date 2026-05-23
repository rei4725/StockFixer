"""ユニットテスト: src.utils.db.migration_runner モジュール"""

import os
import tempfile
import unittest

import duckdb


def _in_memory_con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(":memory:")


class TestDiscoverMigrations(unittest.TestCase):
    """_discover_migrations() のテスト"""

    def test_returns_empty_for_missing_dir(self):
        from src.utils.db.migration_runner import _discover_migrations

        result = _discover_migrations("/nonexistent/path/to/migrations")
        self.assertEqual(result, [])

    def test_excludes_rollback_files(self):
        from src.utils.db.migration_runner import _discover_migrations

        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "0001_initial.sql"), "w").close()
            open(os.path.join(tmpdir, "0001_initial.rollback.sql"), "w").close()
            result = _discover_migrations(tmpdir)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "0001")

    def test_returns_sorted_versions(self):
        from src.utils.db.migration_runner import _discover_migrations

        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "0003_third.sql"), "w").close()
            open(os.path.join(tmpdir, "0001_first.sql"), "w").close()
            open(os.path.join(tmpdir, "0002_second.sql"), "w").close()
            result = _discover_migrations(tmpdir)

        versions = [r[0] for r in result]
        self.assertEqual(versions, ["0001", "0002", "0003"])

    def test_ignores_non_sql_files(self):
        from src.utils.db.migration_runner import _discover_migrations

        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "0001_initial.sql"), "w").close()
            open(os.path.join(tmpdir, "README.md"), "w").close()
            result = _discover_migrations(tmpdir)

        self.assertEqual(len(result), 1)

    def test_description_underscores_replaced_with_spaces(self):
        from src.utils.db.migration_runner import _discover_migrations

        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "0001_add_user_table.sql"), "w").close()
            result = _discover_migrations(tmpdir)

        self.assertEqual(result[0][1], "add user table")


class TestSplitStatements(unittest.TestCase):
    """_split_statements() のテスト"""

    def test_splits_on_semicolon(self):
        from src.utils.db.migration_runner import _split_statements

        sql = "CREATE TABLE a (id INTEGER); CREATE TABLE b (id INTEGER)"
        result = _split_statements(sql)
        self.assertEqual(len(result), 2)

    def test_strips_whitespace(self):
        from src.utils.db.migration_runner import _split_statements

        sql = "  SELECT 1  ;  SELECT 2  "
        result = _split_statements(sql)
        self.assertEqual(result[0], "SELECT 1")
        self.assertEqual(result[1], "SELECT 2")

    def test_excludes_empty_fragments(self):
        from src.utils.db.migration_runner import _split_statements

        sql = "SELECT 1;;SELECT 2;"
        result = _split_statements(sql)
        self.assertEqual(len(result), 2)


class TestRunMigrations(unittest.TestCase):
    """run_migrations() のテスト"""

    def _write_migration(self, tmpdir: str, filename: str, sql: str) -> None:
        with open(os.path.join(tmpdir, filename), "w", encoding="utf-8") as f:
            f.write(sql)

    def test_applies_pending_migration(self):
        from src.utils.db.migration_runner import run_migrations

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_migration(tmpdir, "0001_create_foo.sql", "CREATE TABLE foo (id INTEGER)")
            con = _in_memory_con()
            applied = run_migrations(con, tmpdir)

        self.assertEqual(applied, 1)

    def test_skips_already_applied_migration(self):
        from src.utils.db.migration_runner import run_migrations

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_migration(tmpdir, "0001_create_foo.sql", "CREATE TABLE foo (id INTEGER)")
            con = _in_memory_con()
            run_migrations(con, tmpdir)
            applied_second = run_migrations(con, tmpdir)

        self.assertEqual(applied_second, 0)

    def test_applies_multiple_migrations_in_order(self):
        from src.utils.db.migration_runner import run_migrations

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_migration(tmpdir, "0001_first.sql", "CREATE TABLE t1 (id INTEGER)")
            self._write_migration(
                tmpdir, "0002_second.sql", "ALTER TABLE t1 ADD COLUMN name VARCHAR"
            )
            con = _in_memory_con()
            applied = run_migrations(con, tmpdir)

        self.assertEqual(applied, 2)
        # t1 に name カラムが存在すること
        cols = [
            row[0]
            for row in con.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name='t1'"
            ).fetchall()
        ]
        self.assertIn("name", cols)

    def test_creates_schema_migrations_table(self):
        from src.utils.db.migration_runner import run_migrations

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_migration(tmpdir, "0001_initial.sql", "CREATE TABLE dummy (x INTEGER)")
            con = _in_memory_con()
            run_migrations(con, tmpdir)

        tables = [
            row[0]
            for row in con.execute("SELECT table_name FROM information_schema.tables").fetchall()
        ]
        self.assertIn("schema_migrations", tables)

    def test_records_applied_version_in_schema_migrations(self):
        from src.utils.db.migration_runner import run_migrations

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_migration(tmpdir, "0001_initial.sql", "CREATE TABLE dummy (x INTEGER)")
            con = _in_memory_con()
            run_migrations(con, tmpdir)

        rows = con.execute("SELECT version FROM schema_migrations").fetchall()
        versions = [row[0] for row in rows]
        self.assertIn("0001", versions)

    def test_returns_zero_for_empty_migrations_dir(self):
        from src.utils.db.migration_runner import run_migrations

        with tempfile.TemporaryDirectory() as tmpdir:
            con = _in_memory_con()
            applied = run_migrations(con, tmpdir)

        self.assertEqual(applied, 0)

    def test_does_not_destroy_existing_data(self):
        """既存データが migration 実行後も破壊されないことを確認する"""
        from src.utils.db.migration_runner import run_migrations

        con = _in_memory_con()
        con.execute("CREATE TABLE existing_table (id INTEGER, val VARCHAR)")
        con.execute("INSERT INTO existing_table VALUES (1, 'preserved')")

        with tempfile.TemporaryDirectory() as tmpdir:
            # 既存テーブルに触れないマイグレーションを実行
            self._write_migration(tmpdir, "0001_add_new.sql", "CREATE TABLE new_table (x INTEGER)")
            run_migrations(con, tmpdir)

        row = con.execute("SELECT val FROM existing_table WHERE id=1").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "preserved")

    def test_idempotent_create_table_if_not_exists(self):
        """IF NOT EXISTS DDL を含む migration が既存テーブルを壊さないこと"""
        from src.utils.db.migration_runner import run_migrations

        con = _in_memory_con()
        con.execute("CREATE TABLE foo (id INTEGER)")
        con.execute("INSERT INTO foo VALUES (42)")

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_migration(
                tmpdir, "0001_initial.sql", "CREATE TABLE IF NOT EXISTS foo (id INTEGER)"
            )
            run_migrations(con, tmpdir)

        row = con.execute("SELECT id FROM foo").fetchone()
        self.assertEqual(row[0], 42)


class TestGetAppliedMigrations(unittest.TestCase):
    """get_applied_migrations() のテスト"""

    def test_returns_empty_list_when_no_migrations(self):
        from src.utils.db.migration_runner import get_applied_migrations

        con = _in_memory_con()
        result = get_applied_migrations(con)
        self.assertEqual(result, [])

    def test_returns_applied_migrations_sorted(self):
        from src.utils.db.migration_runner import get_applied_migrations, run_migrations

        with tempfile.TemporaryDirectory() as tmpdir:
            for fname, sql in [
                ("0002_second.sql", "CREATE TABLE b (id INTEGER)"),
                ("0001_first.sql", "CREATE TABLE a (id INTEGER)"),
            ]:
                with open(os.path.join(tmpdir, fname), "w") as f:
                    f.write(sql)
            con = _in_memory_con()
            run_migrations(con, tmpdir)
            result = get_applied_migrations(con)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], "0001")
        self.assertEqual(result[1][0], "0002")

    def test_each_entry_has_version_description_applied_at(self):
        from src.utils.db.migration_runner import get_applied_migrations, run_migrations

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "0001_initial.sql"), "w") as f:
                f.write("CREATE TABLE t (x INTEGER)")
            con = _in_memory_con()
            run_migrations(con, tmpdir)
            result = get_applied_migrations(con)

        self.assertEqual(len(result[0]), 3)
        version, description, applied_at = result[0]
        self.assertEqual(version, "0001")
        self.assertEqual(description, "initial")
        self.assertIsInstance(applied_at, str)
