"""
DuckDB→PostgreSQL 移行後の整合性検証スクリプト

Usage:
    python scripts/verify_postgres_migration.py --duckdb-path data/stockfixer.duckdb

各テーブルのCOUNT(*)、および金額系テーブルはSUM(realized_pnl)等も突き合わせる。
不一致があれば非ゼロ終了する。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# python/scripts/ から直接実行された場合（Usage欄の
# `python scripts/verify_postgres_migration.py` 形式）でも `src` を解決できるよう、
# python/ ルートをsys.pathへ追加する（migrate_to_postgres.py と同じパターン）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb  # noqa: E402
from scripts.migrate_to_postgres import _TABLES, _source_table_exists  # noqa: E402

from src.utils.data_path_utils import get_database_url  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

# テーブル名 → 追加で照合する SUM 対象カラム
_SUM_CHECKS: dict[str, list[str]] = {
    "paper_balance": ["balance"],
    "paper_positions": ["qty"],
    "paper_orders": ["realized_pnl"],
}

# 移行時に意図的に除外した行がある場合、そのテーブルの検証もmigrate_to_postgres.py
# 側と同じ絞り込みで件数を数える（本番切り替え時、prediction_resultsの
# model_version IS NULLな古いレコード705件を除外した。詳細はmigrate_to_postgres.py
# のmigrate_table()コメント参照）。
_EXCLUDED_FILTERS: dict[str, str] = {
    "prediction_results": ' WHERE "model_version" IS NOT NULL',
}


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> object:
    """`fetchone()` は空結果だと None を返しうる（mypy対策）。
    COUNT/SUM は必ず1行返るクエリにのみ用いるため、None時は0扱いにする。
    """
    row = con.execute(sql).fetchone()
    return row[0] if row else 0


def verify_table(src_con: duckdb.DuckDBPyConnection, table: str) -> bool:
    where_clause = _EXCLUDED_FILTERS.get(table, "")
    src_count = _scalar(src_con, f'SELECT COUNT(*) FROM "{table}"{where_clause}')
    pg_count = _scalar(src_con, f'SELECT COUNT(*) FROM pg."{table}"')
    if src_count != pg_count:
        logger.error(f"件数不一致: {table} DuckDB={src_count} Postgres={pg_count}")
        return False

    ok = True
    for col in _SUM_CHECKS.get(table, []):
        src_sum = _scalar(src_con, f'SELECT COALESCE(SUM("{col}"), 0) FROM "{table}"')
        pg_sum = _scalar(src_con, f'SELECT COALESCE(SUM("{col}"), 0) FROM pg."{table}"')
        if abs(float(src_sum) - float(pg_sum)) > 1e-6:  # type: ignore[arg-type]
            logger.error(f"合計値不一致: {table}.{col} DuckDB={src_sum} Postgres={pg_sum}")
            ok = False

    if ok:
        logger.info(f"検証OK: {table} ({src_count}行)")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="移行後の整合性検証")
    parser.add_argument("--duckdb-path", required=True)
    args = parser.parse_args()

    src_con = duckdb.connect(args.duckdb_path, read_only=True)
    src_con.execute("INSTALL postgres")
    src_con.execute("LOAD postgres")
    src_con.execute(f"ATTACH '{get_database_url()}' AS pg (TYPE POSTGRES)")

    all_ok = True
    for table in _TABLES:
        if not _source_table_exists(src_con, table):
            logger.info(f"移行元に存在しないためスキップ: {table}")
            continue
        if not verify_table(src_con, table):
            all_ok = False

    src_con.close()
    if not all_ok:
        logger.error("=== 整合性検証NG。切り替えを中止してください ===")
        return 1
    logger.info("=== 整合性検証OK。切り替え可能です ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
