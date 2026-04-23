"""ユニットテスト: src.utils.db._connection モジュール"""
import unittest
from unittest.mock import MagicMock, patch

import duckdb


class TestCloseConnection(unittest.TestCase):
    """close_connection() のテスト"""

    def test_close_connection_resets_flag(self):
        """close_connection() が _tables_initialized フラグを False にリセットすること"""
        import src.utils.db._connection as mod

        mod._tables_initialized = True
        mod.close_connection()
        self.assertFalse(mod._tables_initialized)

    def test_close_connection_idempotent(self):
        """すでに False の状態で呼んでも例外が出ないこと"""
        import src.utils.db._connection as mod

        mod._tables_initialized = False
        mod.close_connection()  # 例外なし
        self.assertFalse(mod._tables_initialized)


class TestDbConnectionContextManager(unittest.TestCase):
    """_db_connection() コンテキストマネージャーのテスト"""

    def _make_mock_con(self):
        """テーブル初期化が通るように execute を設定したモック接続を返す"""
        mock_con = MagicMock(spec=duckdb.DuckDBPyConnection)
        # paper_balance の COUNT(*) が 1 を返す（INSERT をスキップ）
        mock_con.execute.return_value = MagicMock(
            fetchone=MagicMock(return_value=(1,)),
            fetchall=MagicMock(
                return_value=[("realized_pnl",), ("market",), ("predicted_at",), ("signal_price",)]
            ),
        )
        return mock_con

    def test_yields_connection(self):
        """`with _db_connection() as con:` でモック接続が得られること"""
        import src.utils.db._connection as mod

        mock_con = self._make_mock_con()

        with (
            patch("src.utils.db._connection.ensure_dir"),
            patch("src.utils.db._connection.get_data_dir", return_value="/tmp"),
            patch("src.utils.db._connection.get_db_path", return_value="/tmp/test.db"),
            patch("duckdb.connect", return_value=mock_con),
        ):
            mod._tables_initialized = False
            with mod._db_connection() as yielded_con:
                self.assertIs(yielded_con, mock_con)

        # with ブロック終了後に close() が呼ばれること
        mock_con.close.assert_called_once()

    def test_skips_init_if_already_initialized(self):
        """_tables_initialized=True のときは _init_tables が呼ばれないこと"""
        import src.utils.db._connection as mod

        mock_con = self._make_mock_con()

        with (
            patch("src.utils.db._connection.ensure_dir"),
            patch("src.utils.db._connection.get_data_dir", return_value="/tmp"),
            patch("src.utils.db._connection.get_db_path", return_value="/tmp/test.db"),
            patch("duckdb.connect", return_value=mock_con),
            patch("src.utils.db._connection._init_tables") as mock_init,
        ):
            mod._tables_initialized = True
            with mod._db_connection() as _:
                pass
            mock_init.assert_not_called()

    def test_retries_on_ioerror(self):
        """IOException が出たときにリトライされ、成功後に接続を返すこと"""
        import src.utils.db._connection as mod

        mock_con = self._make_mock_con()
        connect_side_effects = [duckdb.IOException("locked"), mock_con]

        with (
            patch("src.utils.db._connection.ensure_dir"),
            patch("src.utils.db._connection.get_data_dir", return_value="/tmp"),
            patch("src.utils.db._connection.get_db_path", return_value="/tmp/test.db"),
            patch("duckdb.connect", side_effect=connect_side_effects),
            patch("src.utils.db._connection._RETRY_DELAY", 0),
            patch("time.sleep"),
        ):
            mod._tables_initialized = True  # skip _init_tables
            with mod._db_connection() as con:
                self.assertIs(con, mock_con)

    def test_raises_after_all_retries_fail(self):
        """リトライ上限を超えても失敗したら例外が送出されること"""
        import src.utils.db._connection as mod

        with (
            patch("src.utils.db._connection.ensure_dir"),
            patch("src.utils.db._connection.get_data_dir", return_value="/tmp"),
            patch("src.utils.db._connection.get_db_path", return_value="/tmp/test.db"),
            patch("duckdb.connect", side_effect=duckdb.IOException("always locked")),
            patch("src.utils.db._connection._RETRY_COUNT", 2),
            patch("src.utils.db._connection._RETRY_DELAY", 0),
            patch("time.sleep"),
        ):
            with self.assertRaises(duckdb.IOException):
                with mod._db_connection():
                    pass

    def test_filelock_released_on_db_connection_failure(self):
        """全リトライ失敗時（con is None）でも FileLock.release() が呼ばれることを検証"""
        import src.utils.db._connection as mod

        mock_lock_instance = MagicMock()

        with (
            patch("src.utils.db._connection.ensure_dir"),
            patch("src.utils.db._connection.get_data_dir", return_value="/tmp"),
            patch("src.utils.db._connection.get_db_path", return_value="/tmp/test.db"),
            patch("duckdb.connect", side_effect=duckdb.IOException("DB locked")),
            patch("src.utils.db._connection.FileLock", return_value=mock_lock_instance),
        ):
            with self.assertRaises(duckdb.IOException):
                with mod._db_connection():
                    pass

        mock_lock_instance.release.assert_called_once()


class TestGetConnection(unittest.TestCase):
    """get_connection() 非推奨関数のテスト"""

    def test_returns_connection_and_warns(self):
        """get_connection() は DeprecationWarning を出して接続を返すこと"""
        import src.utils.db._connection as mod

        mock_con = MagicMock(spec=duckdb.DuckDBPyConnection)
        mock_con.execute.return_value = MagicMock(
            fetchone=MagicMock(return_value=(1,)),
            fetchall=MagicMock(
                return_value=[("realized_pnl",), ("market",), ("predicted_at",), ("signal_price",)]
            ),
        )

        with (
            patch("src.utils.db._connection.ensure_dir"),
            patch("src.utils.db._connection.get_data_dir", return_value="/tmp"),
            patch("src.utils.db._connection.get_db_path", return_value="/tmp/test.db"),
            patch("duckdb.connect", return_value=mock_con),
        ):
            mod._tables_initialized = True  # skip _init_tables
            import warnings

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                con = mod.get_connection()
            self.assertIs(con, mock_con)
            self.assertTrue(any(issubclass(warning.category, DeprecationWarning) for warning in w))


class TestGetReadonlyConnection(unittest.TestCase):
    """get_readonly_connection() のテスト"""

    def test_returns_readonly_connection(self):
        """get_readonly_connection() が read_only=True で duckdb.connect を呼ぶこと"""
        import src.utils.db._connection as mod

        mock_con = MagicMock(spec=duckdb.DuckDBPyConnection)

        with (
            patch("src.utils.db._connection.ensure_dir"),
            patch("src.utils.db._connection.get_data_dir", return_value="/tmp"),
            patch("src.utils.db._connection.get_db_path", return_value="/tmp/test.db"),
            patch("duckdb.connect", return_value=mock_con) as mock_connect,
        ):
            result = mod.get_readonly_connection()
            mock_connect.assert_called_once_with("/tmp/test.db", read_only=True)
            self.assertIs(result, mock_con)


class TestInitTables(unittest.TestCase):
    """_init_tables() のテスト"""

    def test_creates_tables(self):
        """_init_tables() が con.execute を複数回呼ぶこと（テーブルDDL実行）"""
        import src.utils.db._connection as mod

        mock_con = MagicMock(spec=duckdb.DuckDBPyConnection)
        # COUNT(*) → 1 (paper_balanceに既存データあり)
        # fetchall → 既存カラム一覧（ALTER TABLE がスキップされるよう realized_pnl 等を含む）
        mock_con.execute.return_value = MagicMock(
            fetchone=MagicMock(return_value=(1,)),
            fetchall=MagicMock(
                return_value=[("realized_pnl",), ("market",), ("predicted_at",), ("signal_price",)]
            ),
        )

        mod._init_tables(mock_con)
        self.assertTrue(mock_con.execute.call_count > 0)

    def test_inserts_initial_balance_when_empty(self):
        """paper_balance が空のとき、初期残高 INSERT が呼ばれること"""
        import src.utils.db._connection as mod

        mock_con = MagicMock(spec=duckdb.DuckDBPyConnection)

        call_count = 0

        def mock_execute(sql, *args, **kwargs):
            nonlocal call_count
            result = MagicMock()
            if "COUNT(*)" in sql:
                result.fetchone.return_value = (0,)  # 空
            elif "column_name" in sql:
                result.fetchall.return_value = [
                    ("realized_pnl",),
                    ("market",),
                    ("predicted_at",),
                    ("signal_price",),
                ]
            else:
                result.fetchone.return_value = (1,)
                result.fetchall.return_value = []
            call_count += 1
            return result

        mock_con.execute.side_effect = mock_execute

        mod._init_tables(mock_con)

        # INSERT INTO paper_balance が呼ばれていること
        insert_calls = [
            str(c) for c in mock_con.execute.call_args_list if "INSERT INTO paper_balance" in str(c)
        ]
        self.assertTrue(len(insert_calls) > 0, "paper_balance の初期残高 INSERT が呼ばれていない")


class TestFileLock(unittest.TestCase):
    """_db_connection() の FileLock 排他制御テスト"""

    def _make_mock_con(self):
        mock_con = MagicMock(spec=duckdb.DuckDBPyConnection)
        mock_con.execute.return_value = MagicMock(
            fetchone=MagicMock(return_value=(1,)),
            fetchall=MagicMock(
                return_value=[("realized_pnl",), ("market",), ("predicted_at",), ("signal_price",)]
            ),
        )
        return mock_con

    def test_filelock_acquired_during_db_connection(self):
        """_db_connection() 中に FileLock.acquire() が呼ばれることを検証"""
        import src.utils.db._connection as mod

        mock_con = self._make_mock_con()
        mock_lock_instance = MagicMock()

        with (
            patch("src.utils.db._connection.ensure_dir"),
            patch("src.utils.db._connection.get_data_dir", return_value="/tmp"),
            patch("src.utils.db._connection.get_db_path", return_value="/tmp/test.db"),
            patch("duckdb.connect", return_value=mock_con),
            patch("src.utils.db._connection.FileLock", return_value=mock_lock_instance),
        ):
            mod._tables_initialized = True
            with mod._db_connection():
                pass

        mock_lock_instance.acquire.assert_called_once()

    def test_filelock_released_after_success(self):
        """正常終了後に FileLock.release() が呼ばれることを検証"""
        import src.utils.db._connection as mod

        mock_con = self._make_mock_con()
        mock_lock_instance = MagicMock()

        with (
            patch("src.utils.db._connection.ensure_dir"),
            patch("src.utils.db._connection.get_data_dir", return_value="/tmp"),
            patch("src.utils.db._connection.get_db_path", return_value="/tmp/test.db"),
            patch("duckdb.connect", return_value=mock_con),
            patch("src.utils.db._connection.FileLock", return_value=mock_lock_instance),
        ):
            mod._tables_initialized = True
            with mod._db_connection():
                pass

        mock_lock_instance.release.assert_called_once()

    def test_filelock_released_on_exception(self):
        """例外発生時でも FileLock.release() が呼ばれることを検証（finallyの保証）"""
        import src.utils.db._connection as mod

        mock_con = self._make_mock_con()
        mock_lock_instance = MagicMock()

        with (
            patch("src.utils.db._connection.ensure_dir"),
            patch("src.utils.db._connection.get_data_dir", return_value="/tmp"),
            patch("src.utils.db._connection.get_db_path", return_value="/tmp/test.db"),
            patch("duckdb.connect", return_value=mock_con),
            patch("src.utils.db._connection.FileLock", return_value=mock_lock_instance),
        ):
            mod._tables_initialized = True
            with self.assertRaises(ValueError):
                with mod._db_connection():
                    raise ValueError("テスト用例外")

        mock_lock_instance.release.assert_called_once()

    def test_filelock_timeout_raises_runtime_error(self):
        """FileLockTimeout 発生時に RuntimeError が raise されることを検証"""
        from filelock import Timeout as FileLockTimeout

        import src.utils.db._connection as mod

        mock_lock_instance = MagicMock()
        mock_lock_instance.acquire.side_effect = FileLockTimeout("/tmp/test.db.lock")

        with (
            patch("src.utils.db._connection.ensure_dir"),
            patch("src.utils.db._connection.get_data_dir", return_value="/tmp"),
            patch("src.utils.db._connection.get_db_path", return_value="/tmp/test.db"),
            patch("src.utils.db._connection.FileLock", return_value=mock_lock_instance),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                with mod._db_connection():
                    pass

        self.assertIn("タイムアウト", str(ctx.exception))
