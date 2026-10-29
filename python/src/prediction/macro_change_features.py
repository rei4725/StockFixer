"""
統合モデルの相場全体の指標を、水準ではなく前日比の変化率で特徴量にする（#1374）

水準の列（sp500_close_lag1 など）は日ごとに全銘柄で同じ値をとり、予想が全銘柄で
同じ向きにそろう。学習（challenger）では水準の列を外し、同じ指標の変化率の列を使う。

変化率はほかの特徴量のラグと同じく t の行に t の値を使わない。stock_features の
ラグの列（<指標>_lag<k> は t-k の値）から、<指標>_pct_lag<k> = lag<k> / lag<k+1> - 1
（t-k の日の前日比）を同じ行の中で作る。生の値の列（当日の値）は使わない。
"""

import re

import numpy as np
import pandas as pd

MACRO_LEVEL_COLUMNS = (
    "sp500_close",
    "gold_close",
    "dxy_close",
    "usdjpy_close",
    "tnx_close",
    "vix_close",
)

_LEVEL_PATTERN = re.compile(rf"^(?:{'|'.join(MACRO_LEVEL_COLUMNS)})(?:_lag\d+)?$")


def is_macro_level_column(name: object) -> bool:
    """相場全体の指標の水準の列（生の値かそのラグ）か。"""
    return bool(_LEVEL_PATTERN.match(str(name)))


def add_macro_change_features(X: pd.DataFrame) -> pd.DataFrame:
    """水準のラグの列から前日比の変化率の列を足した写しを返す（列は消さない）。"""
    new_cols: dict = {}
    for name in MACRO_LEVEL_COLUMNS:
        lag = 1
        while f"{name}_lag{lag}" in X.columns and f"{name}_lag{lag + 1}" in X.columns:
            prev = X[f"{name}_lag{lag + 1}"].astype(float)
            change = X[f"{name}_lag{lag}"].astype(float) / prev - 1
            new_cols[f"{name}_pct_lag{lag}"] = change.replace([np.inf, -np.inf], np.nan)
            lag += 1
    if not new_cols:
        return X
    added = pd.DataFrame(new_cols, index=X.index)
    return pd.concat([X.drop(columns=list(added.columns), errors="ignore"), added], axis=1)


def drop_macro_level_columns(X: pd.DataFrame) -> pd.DataFrame:
    """相場全体の指標の水準の列を外す（学習の特徴量の選び方だけに使う）。"""
    return X.drop(columns=[c for c in X.columns if is_macro_level_column(c)])
