"""DuckDB の物理コンパクション（再構築によるファイル容量回収）。

DuckDB の VACUUM はファイルを縮小しないため、肥大化した DB を実際に小さくするには
「生存行だけを新しいファイルへコピー」する再構築が必要。本モジュールは元 DB を
read-only で開き、各テーブルを**型を保持して**新ファイルへコピーする。診断ログ
（shap_values / feature_selection_log）は retention 条件で絞ってコピーする。

注意: 元 DB が他プロセス（コンテナ）に開かれている間は read-only でも開けない
（DuckDB の設定衝突）。実行前にコンテナを停止すること。
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import duckdb
from filelock import FileLock

from src.utils.db.retention import _LOG_TABLES, _cutoff
from src.utils.logger import get_logger

logger = get_logger(__name__)

# コンパクション中は他プロセスの接続を排他するため FileLock を保持する。
# 取得待ちタイムアウト（保持時間ではなく acquire 待ちの上限）。
_COMPACT_LOCK_TIMEOUT = 300.0


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


def swap_compacted(
    db_path: str,
    new_path: str,
    keep_backup: bool = True,
    now: Optional[datetime] = None,
) -> Optional[str]:
    """コンパクション済みファイル new_path を db_path に入れ替える。

    元ファイルを ``.bak-<UTC時刻>`` に退避してから new_path をリネームする。
    ``keep_backup=False`` の場合は退避ファイルを削除する。

    Args:
        db_path: 入れ替え先（現行 DB）のパス。
        new_path: コンパクション済みの新ファイル。
        keep_backup: True なら退避ファイルを残す。False なら削除する。
        now: 退避ファイル名のタイムスタンプ基準（テスト用）。

    Returns:
        退避ファイルのパス。keep_backup=False で削除した場合は None。
    """
    ts = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    bak_path = f"{db_path}.bak-{ts}"
    os.replace(db_path, bak_path)
    os.replace(new_path, db_path)
    if not keep_backup:
        try:
            os.remove(bak_path)
        except OSError as e:
            logger.warning("退避ファイルの削除に失敗: %s (%s)", bak_path, e)
        return None
    return bak_path


def compact_in_place(
    db_path: str,
    retention_days: int,
    keep_backup: bool = False,
    now: Optional[datetime] = None,
) -> dict[str, tuple[int, int]]:
    """db_path を物理コンパクションしてアトミックに入れ替える（同一プロセス内向け）。

    FileLock（``db_path + ".lock"``）を取得して他プロセス／他接続を排他してから、
    ``db_path + ".compact"`` を構築し os.replace で入れ替える。

    前提: 呼び出し時、同一プロセス内に db_path への開いた DuckDB 接続が無いこと
    （短命接続パターンの with ブロックを抜けた後に呼ぶ）。

    Args:
        db_path: 対象 DB ファイル。
        retention_days: 診断ログの保持日数。
        keep_backup: True なら退避ファイル（.bak-*）を残す。
        now: 基準時刻（テスト用）。

    Returns:
        {テーブル名: (元行数, コピー後行数)}
    """
    new_path = db_path + ".compact"
    if os.path.exists(new_path):
        os.remove(new_path)

    lock = FileLock(db_path + ".lock", timeout=_COMPACT_LOCK_TIMEOUT)
    with lock:
        try:
            counts = compact_database(db_path, new_path, retention_days, now=now)
            swap_compacted(db_path, new_path, keep_backup=keep_backup, now=now)
        except Exception:
            # 失敗時は新ファイルを破棄し、元ファイルを温存する
            if os.path.exists(new_path):
                try:
                    os.remove(new_path)
                except OSError:
                    logger.warning("失敗後の新ファイル削除に失敗: %s", new_path)
            raise

    return counts
