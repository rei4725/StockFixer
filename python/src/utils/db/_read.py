"""
pd.read_sql 読み込み結果の dtype 補正ヘルパー（psycopg3版）

ある銘柄のスライスで数値列が全行NULLの場合、pd.read_sql は当該列を
float64 ではなく object dtype（値は None）として返す（DuckDB の `.df()`
変換では起きなかった、psycopg 経由特有の挙動）。この object 列がそのまま
XGBoost 等の厳密な dtype チェックへ渡ると落ちるため、読み込み直後に
数値列へ強制変換する。
"""

from typing import Collection

import pandas as pd


def coerce_object_numeric_columns(df: pd.DataFrame, exclude: Collection[str]) -> pd.DataFrame:
    """object dtype で読み込まれた数値列を float に強制変換する。

    Args:
        df: pd.read_sql の読み込み結果
        exclude: 数値化しない列名（日付・文字列列など）
    """
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
