"""
PostgreSQL メンテナンス（VACUUM）モジュール

DuckDB版は物理ファイルの再構築（ATTACH+コピー+ファイル入れ替え）を行っていたが、
Postgresでは VACUUM (ANALYZE) が対応する肥大化対策になる。
VACUUM はトランザクションブロック内では実行できないため、autocommit接続を使う。
"""

import psycopg

from src.utils.data_path_utils import get_database_url
from src.utils.logger import get_logger

logger = get_logger(__name__)


def vacuum_database() -> None:
    """DB全体に VACUUM (ANALYZE) を実行する。"""
    con = psycopg.connect(get_database_url(), autocommit=True)
    try:
        con.execute("VACUUM (ANALYZE)")
        logger.info("VACUUM (ANALYZE) 完了")
    finally:
        con.close()
