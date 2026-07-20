"""_connection.py のユニットテスト（psycopg版）"""

from unittest.mock import MagicMock, patch

import pytest
from psycopg_pool import PoolTimeout

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
