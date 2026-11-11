"""統合モデルの学習の、日付での並べ替えと学習・検証の分け方（unified_model_pipeline から分けた）。"""

from typing import Optional, Tuple

import pandas as pd

_TRIAL_MODEL_SUFFIXES = ("_challenger", "_candidate")


def is_trial_model(model_name: str) -> bool:
    """challenger と夜間の候補の名前なら True（#1774）。"""
    return model_name.endswith(_TRIAL_MODEL_SUFFIXES)


def sort_by_date(
    X: pd.DataFrame, y: pd.Series, dates: Optional[pd.Series]
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """行を日付の順に並べ（同じ日付の中は今の並び）、index を振り直す（#1363）。

    元の表は market・symbol 順に並ぶため、行の並びのまま末尾を検証にすると、
    末尾の銘柄の全期間が検証になり、その銘柄を学習しない。日付の分からない行は先頭に置く。
    dates が無いときは今の並びのままにする。
    """
    if dates is None:
        return X, y, None
    day = pd.Series(pd.to_datetime(pd.Series(dates).to_numpy(), errors="coerce"))
    order = day.sort_values(kind="stable", na_position="first").index.to_numpy()
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    return X, y, day.iloc[order].reset_index(drop=True)


def split_train_val_by_date(
    X: pd.DataFrame, y: pd.Series, day: Optional[pd.Series]
) -> Tuple[pd.DataFrame, pd.Series, Optional[list]]:
    """日付の順の X を、直近のおよそ 20%（最低 500 行）を検証に、それより前を学習に分ける。

    切れ目は日付の境にそろえ、同じ日付の行が学習と検証にまたがらないようにする
    （切れ目の日付の行はすべて検証に入る）。学習が 1000 行に満たなければ分けない。
    日付が無いときは、行の並びのまま末尾を検証にする（今までの分け方）。
    """
    n_val = max(int(len(X) * 0.2), 500)
    n_train = len(X) - n_val
    if day is not None and 0 < n_train < len(X) and pd.notna(day.iloc[n_train]):
        n_nat = int(day.isna().sum())
        n_train = n_nat + int(day.iloc[n_nat:].searchsorted(day.iloc[n_train], side="left"))
    if n_train < 1000:
        return X, y, None
    X_val, y_val = X.iloc[n_train:], y.iloc[n_train:]
    return X.iloc[:n_train], y.iloc[:n_train], [(X_val, y_val)]
