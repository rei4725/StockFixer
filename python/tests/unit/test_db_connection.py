"""_connection.py のユニットテスト（psycopg版）"""

from unittest.mock import MagicMock, patch

import pytest
from psycopg_pool import PoolTimeout

from src.utils.db import _connection
from src.utils.db._connection import (
    DbLockTimeoutError,
    _db_connection,
    close_connection,
    set_test_connection,
)


class TestDbConnectionTestMode:
    def teardown_method(self):
        set_test_connection(None)

    def test_uses_injected_test_connection_when_set(self):
        fake_con = MagicMock()
        set_test_connection(fake_con)

        with _db_connection() as con:
            assert con is fake_con

        # 同一テスト内の複数回呼び出しでも同じ接続が返ること
        with _db_connection() as con2:
            assert con2 is fake_con

    def test_does_not_touch_pool_when_test_connection_set(self):
        fake_con = MagicMock()
        set_test_connection(fake_con)

        with patch("src.utils.db._connection._get_pool") as mock_get_pool:
            with _db_connection():
                pass
            mock_get_pool.assert_not_called()


class TestDbConnectionPoolTimeout:
    def teardown_method(self):
        close_connection()

    def test_raises_db_lock_timeout_error_on_pool_timeout(self):
        mock_pool = MagicMock()
        mock_pool.connection.side_effect = PoolTimeout("no connection available")

        with patch("src.utils.db._connection._get_pool", return_value=mock_pool):
            with pytest.raises(DbLockTimeoutError):
                with _db_connection(lock_timeout=0.1):
                    pass


class TestDbConnectionMigrationLock:
    def setup_method(self):
        _connection._tables_initialized = False

    def teardown_method(self):
        close_connection()

    def test_wraps_first_time_migration_in_advisory_lock(self):
        mock_con = MagicMock()
        mock_pool = MagicMock()
        mock_pool.connection.return_value.__enter__.return_value = mock_con
        mock_pool.connection.return_value.__exit__.return_value = False

        with patch("src.utils.db._connection._get_pool", return_value=mock_pool), patch(
            "src.utils.db._connection.run_migrations"
        ) as mock_run_migrations:
            manager = MagicMock()
            manager.attach_mock(mock_con.execute, "execute")
            manager.attach_mock(mock_run_migrations, "run_migrations")

            with _db_connection():
                pass

        # 呼び出し順: pg_advisory_lock -> run_migrations -> pg_advisory_unlock
        call_names = [call[0] for call in manager.mock_calls]
        assert call_names == ["execute", "run_migrations", "execute"]

        lock_call, unlock_call = mock_con.execute.call_args_list
        assert lock_call.args[0] == "SELECT pg_advisory_lock(%s)"
        assert unlock_call.args[0] == "SELECT pg_advisory_unlock(%s)"
        mock_run_migrations.assert_called_once_with(mock_con)
        assert _connection._tables_initialized is True

    def test_releases_advisory_lock_even_if_migration_raises(self):
        mock_con = MagicMock()
        mock_pool = MagicMock()
        mock_pool.connection.return_value.__enter__.return_value = mock_con
        mock_pool.connection.return_value.__exit__.return_value = False

        with patch("src.utils.db._connection._get_pool", return_value=mock_pool), patch(
            "src.utils.db._connection.run_migrations",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(RuntimeError):
                with _db_connection():
                    pass

        lock_call, unlock_call = mock_con.execute.call_args_list
        assert lock_call.args[0] == "SELECT pg_advisory_lock(%s)"
        assert unlock_call.args[0] == "SELECT pg_advisory_unlock(%s)"
        # マイグレーションが失敗しているのでフラグは立たない
        assert _connection._tables_initialized is False
