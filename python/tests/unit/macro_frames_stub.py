"""テスト用の相場の指標の表（クロスアセット・追加マクロの決まった列を全部持つ）。

取得の関数が None を返すと特徴量を作らない（#1499）ため、特徴量を作る道を確かめるテストは
この表を返すモックを使う。値は日ごとに少しずつ動かし、派生指標が NaN にならないようにする。
"""

import numpy as np
import pandas as pd

_DATES = pd.date_range("2015-01-01", "2030-12-31", freq="D")
_STEPS = np.arange(len(_DATES), dtype=float)


def cross_asset_frame(*_args: object, **_kwargs: object) -> pd.DataFrame:
    """vix_close・usdjpy_close・tnx_close を全部持つ表を返す。"""
    return pd.DataFrame(
        {
            "vix_close": 15.0 + np.sin(_STEPS / 7.0) * 3.0,
            "usdjpy_close": 140.0 + np.sin(_STEPS / 11.0) * 5.0,
            "tnx_close": 4.0 + np.sin(_STEPS / 13.0) * 0.5,
        },
        index=_DATES,
    )


def additional_macro_frame(*_args: object, **_kwargs: object) -> pd.DataFrame:
    """sp500_close・dxy_close・gold_close を全部持つ表を返す。"""
    return pd.DataFrame(
        {
            "sp500_close": 4000.0 + np.sin(_STEPS / 9.0) * 100.0,
            "dxy_close": 100.0 + np.sin(_STEPS / 17.0) * 2.0,
            "gold_close": 2000.0 + np.sin(_STEPS / 19.0) * 50.0,
        },
        index=_DATES,
    )
