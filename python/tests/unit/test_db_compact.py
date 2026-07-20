"""PostgreSQL メンテナンス（VACUUM）の単体テスト。

- vacuum_database() が VACUUM (ANALYZE) を実行すること
- VACUUM はトランザクションブロック内で実行できないため、autocommit=True の
  新規接続を使っていること
"""

from unittest.mock import MagicMock, patch

from src.utils.db.compact import vacuum_database


def test_vacuum_database_executes_vacuum_analyze():
    mock_con = MagicMock()
    with patch("src.utils.db.compact.psycopg.connect", return_value=mock_con) as mock_connect:
        vacuum_database()

    mock_con.execute.assert_called_once_with("VACUUM (ANALYZE)")
    mock_con.close.assert_called_once()
    # VACUUM はトランザクションブロック内では実行できないため、
    # autocommit=True の接続を使っていることを保証する。
    assert mock_connect.call_args.kwargs.get("autocommit") is True


def test_vacuum_database_closes_connection_even_on_failure():
    mock_con = MagicMock()
    mock_con.execute.side_effect = RuntimeError("boom")
    with patch("src.utils.db.compact.psycopg.connect", return_value=mock_con):
        try:
            vacuum_database()
        except RuntimeError:
            pass
        else:
            raise AssertionError("RuntimeError が伝播しなかった")

    mock_con.close.assert_called_once()
