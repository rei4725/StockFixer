"""DuckDB の物理コンパクション（再構築によるファイル容量回収）。

DuckDB の VACUUM はファイルを縮小しないため、肥大化した DB を実際に小さくするには
「生存行だけを新しいファイルへコピー」する再構築が必要。本モジュールは元 DB を
read-only で開き、各テーブルを**型を保持して**新ファイルへコピーする。診断ログ
（shap_values / feature_selection_log）は retention 条件で絞ってコピーする。

注意: 元 DB が他プロセス（コンテナ）に開かれている間は read-only でも開けない
（DuckDB の設定衝突）。実行前にコンテナを停止すること。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import duckdb

from src.utils.db.retention import _LOG_TABLES, _cutoff
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _src_tables(con: duckdb.DuckDBPyConnection, alias: str) -> list[str]:
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_catalog = ? AND table_schema = 'main' AND table_type = 'BASE TABLE' "
        "ORDER BY table_name",
        [alias],
    ).fetchall()
    return [r[0] for r in rows]


def compact_database(
    src_path: str,
    dst_path: str,
    retention_days: int,
    now: Optional[datetime] = None,
) -> dict[str, tuple[int, int]]:
    """src_path の全テーブルを dst_path へ再構築コピーする。

    診断ログ（_LOG_TABLES）は「直近 retention_days 以内、または各グループの最新
    trained_at」の行のみコピー。それ以外のテーブルは全行コピー。各列の型は元の
    定義を保持する。

    Args:
        src_path: 元 DuckDB ファイル（read-only で開く）。
        dst_path: 出力 DuckDB ファイル（新規作成。既存なら上書き不可なので事前に削除）。
        retention_days: 診断ログの保持日数。
        now: 基準時刻（テスト用）。

    Returns:
        {テーブル名: (元行数, コピー後行数)}
    """
    cutoff = _cutoff(retention_days, now)
    log_keys = {t: keys for t, keys in _LOG_TABLES}
    result: dict[str, tuple[int, int]] = {}

    con = duckdb.connect(dst_path)
    try:
        # ATTACH はパラメータバインド非対応のためパスを直接埋め込む（'' でエスケープ）
        src_escaped = src_path.replace("'", "''")
        con.execute(f"ATTACH '{src_escaped}' AS src (READ_ONLY)")
        for table in _src_tables(con, "src"):
            # 元テーブルの列定義（型保持）で新テーブルを作成
            desc = con.execute(f'DESCRIBE src."{table}"').fetchall()
            coldefs = ", ".join(f'"{row[0]}" {row[1]}' for row in desc)
            con.execute(f'CREATE TABLE "{table}" ({coldefs})')

            if table in log_keys:
                keys = log_keys[table]
                key_join = " AND ".join(f't2.{k} = src."{table}".{k}' for k in keys)
                # keep = 直近 cutoff 以降 OR 自グループの最新 trained_at
                where = (
                    f"trained_at >= ? OR trained_at >= "
                    f'(SELECT MAX(t2.trained_at) FROM src."{table}" t2 WHERE {key_join})'
                )
                con.execute(
                    f'INSERT INTO "{table}" SELECT * FROM src."{table}" WHERE {where}',
                    [cutoff],
                )
            else:
                con.execute(f'INSERT INTO "{table}" SELECT * FROM src."{table}"')

            orig_row = con.execute(f'SELECT COUNT(*) FROM src."{table}"').fetchone()
            new_row = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
            orig_n = int(orig_row[0]) if orig_row else 0
            new_n = int(new_row[0]) if new_row else 0
            result[table] = (orig_n, new_n)
            logger.info("compact: %s %d -> %d 行", table, orig_n, new_n)

        con.execute("DETACH src")
        con.execute("CHECKPOINT")
    finally:
        con.close()

    return result
