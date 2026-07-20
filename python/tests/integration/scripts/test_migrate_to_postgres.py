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


# stock_features に本番相当で存在しうる型のバリエーションを一通りカバーする
# 動的追加列（すべてPostgres移行先baselineには無い）。
# `_duckdb_type_to_postgres` の分岐（整数/浮動小数/真偽値/日時/文字列）を
# それぞれ踏むように選定している。
_DYNAMIC_COLUMNS_MULTI_TYPE = [
    "macd_signal",
    "volume_lag",
    "is_gap_up",
    "last_earnings_ts",
    "listed_date",
    "sector_label",
]


@pytest.fixture
def sample_duckdb_with_uninstalled_dynamic_column(tmp_path):
    """Postgres移行先にはまだ存在しない動的追加列群を持つDuckDBを作る。

    `rsi_14` 用フィクスチャ（上記）とは異なり、こちらは *事前にPostgres側へ
    ALTERしない*。これは移行スクリプトが実際に稼働する状況（Postgresベースライン
    適用直後・アプリがまだ一度もPostgresへ書き込んでいない状態）そのものを再現する。
    列は1種類（DOUBLE）だけでなく、BIGINT/BOOLEAN/TIMESTAMP/DATE/VARCHARの
    バリエーションを持たせ、`_duckdb_type_to_postgres` の各分岐と、
    型ごとのINSERT時キャストが実際に成功することの両方を検証する。
    """
    db_path = str(tmp_path / "sample_stock_features_missing_target_col.duckdb")
    con = duckdb.connect(db_path)
    con.execute(
        "CREATE TABLE stock_features ("
        "market VARCHAR, symbol VARCHAR, row_num INTEGER, "
        "macd_signal DOUBLE, volume_lag BIGINT, is_gap_up BOOLEAN, "
        "last_earnings_ts TIMESTAMP, listed_date DATE, sector_label VARCHAR)"
    )
    con.execute(
        "INSERT INTO stock_features VALUES ("
        "'us', 'AAPL', 0, "
        "12.34, 123456789, TRUE, "
        "TIMESTAMP '2024-01-15 09:30:00', DATE '2020-06-01', 'Technology')"
    )
    con.close()
    return db_path


@pytest.fixture
def attached_src_con_stock_features_missing_target_column(
    sample_duckdb_with_uninstalled_dynamic_column,
):
    """本番のギャップをそのまま再現するfixture: 動的追加列群をPostgres側へ
    事前ALTERしない（テスト開始前に本当に存在しないことも確認する）。

    Postgres側の `stock_features` はTask 2のベースライン（market/symbol/row_num
    の3列）のままであり、`migrate_table` 自身が不足カラムを補うことを検証する。
    """
    with psycopg.connect(get_database_url(), autocommit=True) as check_con:
        placeholders = ", ".join(["%s"] * len(_DYNAMIC_COLUMNS_MULTI_TYPE))
        existing = check_con.execute(
            "SELECT column_name FROM information_schema.columns "
            f"WHERE table_name = 'stock_features' AND column_name IN ({placeholders})",
            _DYNAMIC_COLUMNS_MULTI_TYPE,
        ).fetchall()
        assert existing == [], (
            f"テスト前提が崩れている: {existing} が移行実行前からPostgresに存在する"
            "（本番ギャップの再現になっていない）"
        )

    src_con = duckdb.connect(sample_duckdb_with_uninstalled_dynamic_column, read_only=True)
    src_con.execute("INSTALL postgres")
    src_con.execute("LOAD postgres")
    src_con.execute(f"ATTACH '{get_database_url()}' AS pg (TYPE POSTGRES, READ_ONLY FALSE)")
    try:
        yield src_con
    finally:
        src_con.execute('TRUNCATE TABLE pg."stock_features"')
        src_con.close()
        with psycopg.connect(get_database_url(), autocommit=True) as cleanup_con:
            for col in _DYNAMIC_COLUMNS_MULTI_TYPE:
                cleanup_con.execute(f'ALTER TABLE stock_features DROP COLUMN IF EXISTS "{col}"')


def test_migrate_table_adds_missing_dynamic_column_to_postgres_target(
    attached_src_con_stock_features_missing_target_column, _isolate_db
):
    """本番相当のギャップ（Postgres側にstock_featuresの動的列が一つも無い状態）を
    再現し、`migrate_table` が (a) 例外を出さず、(b) 型の異なる不足カラム群を
    Postgres側へ妥当な型でALTER追加し、(c) それぞれのデータが実際に正しく
    着地することを検証する。
    """
    src_con = attached_src_con_stock_features_missing_target_column

    columns = _get_dynamic_columns(src_con, "stock_features")
    assert columns == ["market", "symbol", "row_num"] + _DYNAMIC_COLUMNS_MULTI_TYPE

    count = migrate_table(src_con, "stock_features", columns)

    # (b) 不足カラムがPostgres側に追加され、型カテゴリごとに妥当であること
    col_types = dict(
        src_con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_catalog = 'pg' AND table_name = 'stock_features' "
            f"AND column_name IN ({', '.join('?' * len(_DYNAMIC_COLUMNS_MULTI_TYPE))})",
            _DYNAMIC_COLUMNS_MULTI_TYPE,
        ).fetchall()
    )
    assert "DOUBLE" in col_types["macd_signal"].upper()
    assert "BIGINT" in col_types["volume_lag"].upper()
    assert "BOOLEAN" in col_types["is_gap_up"].upper()
    assert "TIMESTAMP" in col_types["last_earnings_ts"].upper()
    assert "TIMESTAMP" in col_types["listed_date"].upper()  # DATE -> TIMESTAMP へマップ
    assert "VARCHAR" in col_types["sector_label"].upper()

    # (c) データが正しく着地している
    target_count = src_con.execute('SELECT COUNT(*) FROM pg."stock_features"').fetchone()[0]
    row = src_con.execute(
        'SELECT "market", "symbol", "row_num", "macd_signal", "volume_lag", "is_gap_up", '
        '"last_earnings_ts", "listed_date", "sector_label" FROM pg."stock_features"'
    ).fetchone()

    assert count == 1
    assert target_count == 1
    assert row[0:4] == ("us", "AAPL", 0, 12.34)
    assert row[4] == 123456789
    assert row[5] is True
    assert str(row[6]) == "2024-01-15 09:30:00"
    # DATEソースはTIMESTAMPカラムへ着地するため、時刻部分は0時0分になる
    assert str(row[7]).startswith("2020-06-01")
    assert row[8] == "Technology"
