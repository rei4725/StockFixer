"""
DataFrame一括書き込みヘルパー（psycopg3版）

DuckDBの `con.register()` + `INSERT ... SELECT * FROM <df>` に相当する処理を
psycopg の COPY プロトコルで再現する。

table / columns / key_cols は必ずハードコードされたリテラル値を渡すこと
（ユーザー入力を直接埋め込まない）。
"""

from typing import Optional, Sequence

import pandas as pd
import psycopg


def _quoted_cols(cols: Sequence[str]) -> str:
    return ", ".join(f'"{c}"' for c in cols)


def _prepare_rows(df: pd.DataFrame, cols: Sequence[str]):
    clean = df[list(cols)].astype(object).where(pd.notnull(df[list(cols)]), None)
    return clean.itertuples(index=False, name=None)


def bulk_insert(
    con: psycopg.Connection,
    table: str,
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
) -> None:
    """
    df の全行を table へ直接 COPY する（既存行との重複解決は行わない）。
    呼び出し側が事前に DELETE 済みのケース（stock_features 等）で使う。
    """
    if df.empty:
        return
    cols = list(columns) if columns is not None else list(df.columns)
    col_list = _quoted_cols(cols)

    with con.cursor() as cur:
        with cur.copy(f'COPY "{table}" ({col_list}) FROM STDIN') as copy:
            for row in _prepare_rows(df, cols):
                copy.write_row(row)


def bulk_upsert(
    con: psycopg.Connection,
    table: str,
    df: pd.DataFrame,
    key_cols: Sequence[str],
    columns: Optional[Sequence[str]] = None,
) -> None:
    """
    df の全行を table に upsert する（key_cols が既存行と一致すれば上書き）。
    DuckDB の `INSERT OR REPLACE` 相当。COPYで一時テーブルに流し込み、
    `INSERT ... ON CONFLICT (key_cols) DO UPDATE` で本テーブルへ反映する。
    """
    if df.empty:
        return
    cols = list(columns) if columns is not None else list(df.columns)
    col_list = _quoted_cols(cols)
    key_list = _quoted_cols(key_cols)
    update_cols = [c for c in cols if c not in key_cols]

    with con.cursor() as cur:
        # ON COMMIT DROP はautocommit=True接続では各文が即コミットされるため
        # CREATE直後に一時テーブルが消えてしまう。また同一セッション内で
        # 別テーブルへ再度呼び出された場合、コミットが発生しない限り
        # 前回分が残り DuplicateTable になる。そのため ON COMMIT 節は使わず、
        # 明示的な DROP で寿命を管理する。
        cur.execute("DROP TABLE IF EXISTS _bulk_upsert")
        cur.execute(
            f'CREATE TEMP TABLE _bulk_upsert AS SELECT {col_list} FROM "{table}" WITH NO DATA'
        )
        with cur.copy(f"COPY _bulk_upsert ({col_list}) FROM STDIN") as copy:
            for row in _prepare_rows(df, cols):
                copy.write_row(row)

        if update_cols:
            set_clause = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)
            conflict_action = f"DO UPDATE SET {set_clause}"
        else:
            conflict_action = "DO NOTHING"

        cur.execute(
            f'INSERT INTO "{table}" ({col_list}) '
            f"SELECT {col_list} FROM _bulk_upsert "
            f"ON CONFLICT ({key_list}) {conflict_action}"
        )
        cur.execute("DROP TABLE IF EXISTS _bulk_upsert")
