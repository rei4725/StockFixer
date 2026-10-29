"""
予想に使う最終日の行のラグ特徴量を作る。

pipeline.py からの責務分離（ファイル行数ガード対策）。
pipeline.py から _build_latest_feature_row の名前で re-export する。
"""

from typing import Optional

import pandas as pd


def build_latest_feature_row(df: pd.DataFrame, lag_columns: list) -> Optional[pd.DataFrame]:
    """
    df の最終日の行のラグ特徴量を、create_basic_lag_features と同じ規則で作る。

    create_basic_lag_features は target（翌日の変化率）が NaN の行を落とすため、
    最終日の行を返さない。ここではその行だけを作る。target 以外の欠損と
    earnings_flag の扱いは create_basic_lag_features に揃える。

    Args:
        df: create_basic_lag_features に渡したのと同じ DataFrame
        lag_columns: create_basic_lag_features が返した X の列名（"<列>_lag<N>"）

    Returns:
        最終日1行の DataFrame。作れない（欠損・決算近傍）なら None。
    """
    if df.empty or not lag_columns:
        return None
    last = df.iloc[-1]
    if last.isnull().any():
        return None
    if "earnings_flag" in df.columns and last["earnings_flag"] != 0:
        return None

    values = {}
    for name in lag_columns:
        col, _, lag = str(name).rpartition("_lag")
        if col not in df.columns or not lag.isdigit() or int(lag) >= len(df):
            return None
        values[name] = df[col].iloc[-1 - int(lag)]
    row = pd.DataFrame([values], index=df.index[-1:], columns=lag_columns)
    if row.isnull().any().any():
        return None
    return row
