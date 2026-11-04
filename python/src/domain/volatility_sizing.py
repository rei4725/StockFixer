"""値動きの大きさの見込みと、それに合わせた発注の量・損切りの線を計算する純粋なルール関数（#1528）。

backtest と trading の両 BC から比べに使えるよう domain に置く。既定は off で、どの発注の道からも
まだ呼んでいない。on にするかは、バックテストの比べの数字が出てから決める。

見込みは、過去20日の日次リターンの標準偏差をそのまま使う基準である（#1526 の下調べで効きを測った
ものと同じ定義）。学習したモデルなどの別の見込みを足すのは、この基準より良い場合だけとする。

値が足りない銘柄の既定の扱い: 見込みは None になり、量は補正せず（倍率 1.0）、損切りの線は
渡された固定の率のままにする。荒いかどうか分からない銘柄を、見込みなしで増やしも減らしもしない。

domain は config を import しないため、1銘柄の上限の率は呼び出し側が渡す。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolatilitySizingParams:
    """値動きの補正の切り替えと既定値。

    既定の数値は比べの出発点で、成績で選んだ値ではない。

    Attributes:
        enabled: 補正の切り替え。False なら量も損切りの線も今のまま。
        volatility_window: 標準偏差を取る窓の長さ（日次リターンの数）。
        min_observations: 窓の中で見込みを出すのに要る日次リターンの数（窓の長さ以下）。
        target_daily_volatility: 倍率が 1.0 になる1日の値動き（率）。
        min_scale: 量の倍率の下限。
        max_scale: 量の倍率の上限。
        stop_volatility_multiple: 損切りの線を1日の見込みの何倍にするか（k）。
        max_stop_loss_pct: 損切りの線の上限（率）。1 以上の線は買い持ちで発動しないため抑える。
    """

    enabled: bool = False
    volatility_window: int = 20
    min_observations: int = 20
    target_daily_volatility: float = 0.02
    min_scale: float = 0.25
    max_scale: float = 2.0
    stop_volatility_multiple: float = 3.0
    max_stop_loss_pct: float = 0.30

    def __post_init__(self) -> None:
        if self.volatility_window < 2:
            raise ValueError(f"volatility_window は 2 以上: {self.volatility_window}")
        if not 1 <= self.min_observations <= self.volatility_window:
            raise ValueError(f"min_observations は 1 以上で窓の長さ以下: {self.min_observations}")
        if self.target_daily_volatility <= 0.0:
            raise ValueError(f"target_daily_volatility は正: {self.target_daily_volatility}")
        if not 0.0 < self.min_scale <= self.max_scale:
            raise ValueError(f"倍率の範囲が不正: {self.min_scale}〜{self.max_scale}")
        if self.stop_volatility_multiple <= 0.0:
            raise ValueError(f"stop_volatility_multiple は正: {self.stop_volatility_multiple}")
        if not 0.0 < self.max_stop_loss_pct < 1.0:
            raise ValueError(f"max_stop_loss_pct は 0 と 1 の間: {self.max_stop_loss_pct}")


DEFAULT_VOLATILITY_SIZING = VolatilitySizingParams()


def forecast_volatility_series(
    closes: Union[pd.Series, Sequence[float]],
    *,
    params: VolatilitySizingParams = DEFAULT_VOLATILITY_SIZING,
) -> pd.Series:
    """各時点の1日の値動きの見込み（過去の窓の日次対数リターンの標本標準偏差）を返す。

    t の値は t までの終値だけで作る（後ろ向きの計算だけで先読みが無い）。t の終値で約定する比べには、
    1日ずらして使う。標準偏差は窓の中の平均を引く標本の式（ddof=1）で計算する。

    0以下・NaN・無限の終値は欠けとして扱い、その前後のリターンを作らない。窓の長さはリターンの
    時点の数で数え、欠けを飛ばして窓を過去へ広げない。既定は窓の長さと要る数が同じ 20 なので、
    窓の中に欠けが1つでもある時点は NaN になる（終値の欠けは2つのリターンを消すため、欠けが窓を
    抜けるまで21時点が NaN）。min_observations を窓の長さより小さくすると、窓の中の残りの
    リターンが足りる時点は、それらだけで計算する。

    Args:
        closes: 終値。古い順に並べる。
        params: 窓の長さと要る観測数を読む。

    Returns:
        closes と同じ長さの Series。観測が足りない時点は NaN。
    """
    series = pd.Series(closes, dtype=float)
    valid = series.where(np.isfinite(series) & (series > 0.0))
    log_returns = pd.Series(np.log(valid / valid.shift(1)), index=series.index)
    return log_returns.rolling(
        window=params.volatility_window, min_periods=params.min_observations
    ).std(ddof=1)


def forecast_volatility(
    closes: Union[pd.Series, Sequence[float]],
    *,
    params: VolatilitySizingParams = DEFAULT_VOLATILITY_SIZING,
) -> Optional[float]:
    """渡された終値の最後の時点の見込みを返す。

    値が足りない銘柄、または見込みが有限の正の数にならない銘柄は None を返す。
    None を量と損切りの線の関数へ渡すと、モジュールの docstring の既定の扱いになる。
    """
    series = forecast_volatility_series(closes, params=params)
    if series.empty:
        return None
    value = float(series.iloc[-1])
    return value if _is_usable(value) else None


def volatility_position_scale(
    forecast: Optional[float],
    *,
    params: VolatilitySizingParams = DEFAULT_VOLATILITY_SIZING,
) -> float:
    """量の倍率を返す。見込みに反比例し、min_scale〜max_scale に収める。

    off のとき、または見込みが使えないときは 1.0。
    """
    if not params.enabled or forecast is None or not _is_usable(forecast):
        return 1.0
    raw_scale = params.target_daily_volatility / forecast
    return max(params.min_scale, min(raw_scale, params.max_scale))


def volatility_adjusted_amount(
    base_amount: float,
    balance: float,
    forecast: Optional[float],
    *,
    max_position_rate: float,
    params: VolatilitySizingParams = DEFAULT_VOLATILITY_SIZING,
) -> float:
    """発注の金額に量の倍率を掛け、balance × max_position_rate までに収める。

    off のときは base_amount をそのまま返す。on で見込みが使えないときは倍率 1.0 で上限だけ掛ける。
    """
    if not params.enabled:
        return base_amount
    scaled = base_amount * volatility_position_scale(forecast, params=params)
    return max(0.0, min(scaled, balance * max_position_rate))


def volatility_stop_loss_pct(
    fixed_stop_loss_pct: Optional[float],
    forecast: Optional[float],
    *,
    params: VolatilitySizingParams = DEFAULT_VOLATILITY_SIZING,
) -> Optional[float]:
    """損切りの線（率）を返す。見込みの k 倍で、max_stop_loss_pct までに収める。

    off のとき、または見込みが使えないときは fixed_stop_loss_pct をそのまま返す（None も None のまま）。
    """
    if not params.enabled or forecast is None or not _is_usable(forecast):
        return fixed_stop_loss_pct
    return min(forecast * params.stop_volatility_multiple, params.max_stop_loss_pct)


def _is_usable(forecast: float) -> bool:
    return math.isfinite(forecast) and forecast > 0.0
