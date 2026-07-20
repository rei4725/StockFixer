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
import psycopg
import pytest
from scripts.migrate_to_postgres import _get_dynamic_columns, migrate_table

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


@pytest.fixture
def sample_duckdb_with_dynamic_stock_features(tmp_path):
    """`stock_features` にPostgresベースライン（market/symbol/row_num の3列）
    には無い動的追加列（rsi_14）を持つDuckDBファイルを作る。

    `_get_dynamic_columns` はまさにこの「アプリ側でALTER TABLE ADD COLUMNされた
    動的列」を実行時に検出するために存在するので、baseline の3列だけでは
    このテーブルが特別扱いされている理由を検証できない。
    """
    db_path = str(tmp_path / "sample_stock_features.duckdb")
    con = duckdb.connect(db_path)
    con.execute(
        "CREATE TABLE stock_features ("
        "market VARCHAR, symbol VARCHAR, row_num INTEGER, rsi_14 DOUBLE)"
    )
    con.execute("INSERT INTO stock_features VALUES ('jp', '7203', 1, 55.5)")
    con.close()
    return db_path


@pytest.fixture
def attached_src_con_stock_features(sample_duckdb_with_dynamic_stock_features):
    """`stock_features` を持つDuckDBにPostgres拡張をロードしてATTACHした接続。

    `main()` の実際の呼び出し順（ATTACH → `_get_dynamic_columns` 呼び出し）
    を再現するため、ATTACHをこのfixtureの中で行ってから渡す。

    Postgres側の `stock_features` はベースラインでは market/symbol/row_num の
    3列しか持たない。実運用では特徴量エンジニアリング側の `_ensure_columns`
    （`src/utils/db/stock_features.py`）が動的にALTERして列を追加する。
    このテストは「移行スクリプトのカタログ重複バグ」だけを検証対象にしたい
    ので、その前提（Postgres側にも同名の動的列が既に存在する）を直接
    `ALTER TABLE` で再現してから移行を実行し、テスト後に列を削除して
    ベースライン状態へ戻す。
    """
    with psycopg.connect(get_database_url(), autocommit=True) as setup_con:
        setup_con.execute(
            'ALTER TABLE stock_features ADD COLUMN IF NOT EXISTS "rsi_14" DOUBLE PRECISION'
        )

    src_con = duckdb.connect(sample_duckdb_with_dynamic_stock_features, read_only=True)
    src_con.execute("INSTALL postgres")
    src_con.execute("LOAD postgres")
    src_con.execute(f"ATTACH '{get_database_url()}' AS pg (TYPE POSTGRES, READ_ONLY FALSE)")
    try:
        yield src_con
    finally:
        src_con.execute('TRUNCATE TABLE pg."stock_features"')
        src_con.close()
        with psycopg.connect(get_database_url(), autocommit=True) as cleanup_con:
            cleanup_con.execute('ALTER TABLE stock_features DROP COLUMN IF EXISTS "rsi_14"')


def test_get_dynamic_columns_excludes_attached_postgres_catalog(
    attached_src_con_stock_features, _isolate_db
):
    """回帰テスト: `_get_dynamic_columns` はATTACH後に呼ばれても、移行元DuckDB
    自身のカラムだけを返さねばならない。

    Postgres側ベースラインにも `stock_features`（market/symbol/row_num の3列）
    が存在するため、`table_catalog` で絞り込まずにこのテーブルへ問い合わせると
    移行元(4列: market/symbol/row_num/rsi_14)と移行先(3列)の両方のカラム名が
    連結されて返り、`market`等が重複してしまう
    （移行時に `Duplicate column name "market" in INSERT` で落ちる原因）。
    """
    src_con = attached_src_con_stock_features

    columns = _get_dynamic_columns(src_con, "stock_features")

    assert columns == ["market", "symbol", "row_num", "rsi_14"]


def test_migrate_table_dynamic_stock_features_columns_land_in_postgres(
    attached_src_con_stock_features, _isolate_db
):
    """`_get_dynamic_columns` → `migrate_table` の実際の連携（`main()` と同じ
    呼び出し順）で、動的追加列(rsi_14)を含めて実データがPostgresへ正しく
    移行されることを検証する。例外が出ないことだけでなく、実際の値も確認する。
    """
    src_con = attached_src_con_stock_features

    columns = _get_dynamic_columns(src_con, "stock_features")
    count = migrate_table(src_con, "stock_features", columns)

    target_count = src_con.execute('SELECT COUNT(*) FROM pg."stock_features"').fetchone()[0]
    row = src_con.execute(
        'SELECT "market", "symbol", "row_num", "rsi_14" FROM pg."stock_features"'
    ).fetchone()

    assert count == 1
    assert target_count == 1
    assert row == ("jp", "7203", 1, 55.5)
