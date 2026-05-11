"""
DB マイグレーションランナー

src/utils/db/migrations/ ディレクトリの連番 SQL ファイルを
schema_migrations テーブルで管理しながら適用する。

命名規則:
  NNNN_description.sql          (フォワードマイグレーション)
  NNNN_description.rollback.sql (ロールバック用)
"""

import os
import re
from typing import List, Tuple

import duckdb

from src.utils.logger import get_logger

logger = get_logger(__name__)

_MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), "migrations")
_VERSION_RE = re.compile(r"^(\d{4})_(.+?)\.sql$")


def _ensure_schema_migrations(con: duckdb.DuckDBPyConnection) -> None:
    """schema_migrations テーブルが存在しなければ作成する。"""
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version     VARCHAR NOT NULL PRIMARY KEY,
            description VARCHAR NOT NULL,
            applied_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def _get_applied_versions(con: duckdb.DuckDBPyConnection) -> set:
    rows = con.execute("SELECT version FROM schema_migrations").fetchall()
    return {str(row[0]) for row in rows}


def _discover_migrations(migrations_dir: str) -> List[Tuple[str, str, str]]:
    """
    フォワードマイグレーション SQL ファイルを (version, description, path) のリストで返す（昇順）。
    *.rollback.sql は除外する。
    """
    if not os.path.isdir(migrations_dir):
        return []
    result = []
    for fname in sorted(os.listdir(migrations_dir)):
        if fname.endswith(".rollback.sql"):
            continue
        m = _VERSION_RE.match(fname)
        if m:
            version = m.group(1)
            description = m.group(2).replace("_", " ")
            path = os.path.join(migrations_dir, fname)
            result.append((version, description, path))
    return result


def _split_statements(sql: str) -> List[str]:
    """セミコロンで SQL ステートメントを分割する。空白のみの断片は除外する。"""
    return [s.strip() for s in sql.split(";") if s.strip()]


def run_migrations(
    con: duckdb.DuckDBPyConnection,
    migrations_dir: str = _MIGRATIONS_DIR,
) -> int:
    """
    未適用のマイグレーションを昇順に実行する。

    Returns:
        適用したマイグレーション数
    """
    _ensure_schema_migrations(con)
    applied = _get_applied_versions(con)
    pending = [
        (v, desc, path)
        for v, desc, path in _discover_migrations(migrations_dir)
        if v not in applied
    ]
    for version, description, path in pending:
        logger.info("マイグレーション適用: %s %s", version, description)
        with open(path, encoding="utf-8") as f:
            sql = f.read()
        for statement in _split_statements(sql):
            con.execute(statement)
        con.execute(
            "INSERT INTO schema_migrations (version, description) VALUES (?, ?)",
            [version, description],
        )
        logger.info("マイグレーション完了: %s", version)
    return len(pending)


def get_applied_migrations(con: duckdb.DuckDBPyConnection) -> List[Tuple[str, str, str]]:
    """
    適用済みマイグレーションを (version, description, applied_at) のリストで返す。
    """
    _ensure_schema_migrations(con)
    rows = con.execute(
        "SELECT version, description, applied_at FROM schema_migrations ORDER BY version"
    ).fetchall()
    return [(str(row[0]), str(row[1]), str(row[2])) for row in rows]
