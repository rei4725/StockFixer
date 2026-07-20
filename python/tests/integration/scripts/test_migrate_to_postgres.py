"""migrate_to_postgres.py の統合テスト（実DuckDB・実Postgres両方を使用）。

重要: `migrate_table` はDuckDBの `ATTACH ... (TYPE POSTGRES)` 経由で移行先Postgres
に直接書き込む。これは `_isolate_db`（tests/integration/conftest.py）が
`set_test_connection` で差し替える psycopg 接続とは別物の libpq 接続であり、
`_isolate_db` のロールバックの対象にならない。つまりこのテストはPostgresの
`system_config` テーブルに実際に COMMIT する。そのためテスト自身が
`finally` で対象行を後始末する（使い捨て/CI用Postgresでの実行を前提とする。
本番データが入っている `system_config` に対しては実行しないこと）。
"""

import duckdb
import pytest
from scripts.migrate_to_postgres import migrate_table

from src.utils.data_path_utils import get_database_url


@pytest.fixture
def sample_duckdb(tmp_path):
    db_path = str(tmp_path / "sample.duckdb")
    con = duckdb.connect(db_path)
    con.execute(
        "CREATE TABLE system_config (key VARCHAR PRIMARY KEY, value VARCHAR, updated_at TIMESTAMP)"
    )
    con.execute("INSERT INTO system_config VALUES ('k1', 'v1', CURRENT_TIMESTAMP)")
    con.close()
    return db_path


@pytest.fixture
def attached_src_con(sample_duckdb):
    """sample_duckdb にPostgres拡張をロードしてATTACHしたDuckDB接続。

    ATTACH先への書き込みは `_isolate_db` のロールバック対象外の別接続なので、
    テスト終了時に必ず対象テーブルをTRUNCATEして後始末する。
    """
    src_con = duckdb.connect(sample_duckdb, read_only=True)
    src_con.execute("INSTALL postgres")
    src_con.execute("LOAD postgres")
    # read_only=True はデフォルトでATTACH先にも継承されるため、移行先Postgres
    # へは明示的に READ_ONLY FALSE を指定する（migrate_to_postgres.py と同様）。
    src_con.execute(f"ATTACH '{get_database_url()}' AS pg (TYPE POSTGRES, READ_ONLY FALSE)")
    try:
        yield src_con
    finally:
        # migrate_table 自体がPostgresへ直接COMMITするため、_isolate_db の
        # ロールバックでは消えない。テスト側で明示的に後始末する。
        src_con.execute('TRUNCATE TABLE pg."system_config"')
        src_con.close()


def test_migrate_table_copies_all_rows(attached_src_con, _isolate_db):
    src_con = attached_src_con

    count = migrate_table(src_con, "system_config", ["key", "value", "updated_at"])

    # migrate_table の戻り値（DuckDB側の件数）だけでなく、Postgres側に実際に
    # 行が着地したことも検証する。
    target_count = src_con.execute('SELECT COUNT(*) FROM pg."system_config"').fetchone()[0]
    row = src_con.execute('SELECT "key", "value" FROM pg."system_config"').fetchone()

    assert count == 1
    assert target_count == 1
    assert row == ("k1", "v1")


def test_migrate_table_is_idempotent(attached_src_con, _isolate_db):
    """TRUNCATE→INSERT方式のため、複数回実行しても最終件数は変わらないはず。"""
    src_con = attached_src_con

    first = migrate_table(src_con, "system_config", ["key", "value", "updated_at"])
    second = migrate_table(src_con, "system_config", ["key", "value", "updated_at"])

    target_count = src_con.execute('SELECT COUNT(*) FROM pg."system_config"').fetchone()[0]

    assert first == 1
    assert second == 1
    assert target_count == 1
