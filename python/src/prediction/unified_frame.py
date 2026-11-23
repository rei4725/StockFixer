"""統合モデルの学習の大きな表を、写しを重ねずに選ぶ道具（unified_model_pipeline から分けた、#2134）。"""

from itertools import groupby

import numpy as np
import pandas as pd

# NaN の確かめで1度に写す列の数
_VALID_CHUNK_COLUMNS = 64


def rows_without_nan(df: pd.DataFrame, columns: list) -> pd.Series:
    """columns のどれも NaN でない行を True にする。列のまとまりごとに見て、全列の写しを作らない。"""
    valid = pd.Series(True, index=df.index)
    for start in range(0, len(columns), _VALID_CHUNK_COLUMNS):
        valid &= df[columns[start : start + _VALID_CHUNK_COLUMNS]].notna().all(axis=1)
    return valid


def select_rows_and_columns(df: pd.DataFrame, rows: pd.Series, columns: list) -> pd.DataFrame:
    """df[columns][rows] と同じ表を、返す表のほかに大きな写しを作らずに作る。

    行と列を順に選ぶと、途中の写しがもう1枚できる。並びの続く同じ型の列を1つの配列に
    直に書き込み、その配列をそのまま表にする。
    """
    index = df[[]][rows].index
    mask = rows.to_numpy(dtype=bool)
    parts = []
    for dtype, group in groupby(columns, key=lambda c: df[c].dtype):
        names = list(group)
        if not isinstance(dtype, np.dtype):
            parts.append(df[names][rows])
            continue
        block = np.empty((len(names), len(index)), dtype=dtype)
        for i, name in enumerate(names):
            np.compress(mask, df[name].to_numpy(), out=block[i])
        parts.append(pd.DataFrame(block.T, index=index, columns=names, copy=False))
    if not parts:
        return df[columns][rows]
    return pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]
