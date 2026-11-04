"""バックテストの値動きの見込みの量と損切りの線（#1547）。

計算は domain/volatility_sizing.py の部品に任せ、ここでは「その足で使える見込み」を渡す。
見込みは1日ずらす: 足 t の約定には t-1 の終値までの見込みを使う。翌バー始値の約定でも、同バーの
終値の約定でも、t の終値を見ない。見込みが出ない足は部品の既定の扱い（倍率 1.0・固定の線）に倒れる。

発注の道からは呼ばない。off（None か enabled=False）なら何も計算せず、渡された値をそのまま返す。
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from src.backtest.execution import ExecutionModel
from src.backtest.position_sizing import cap_to_cash
from src.domain.volatility_sizing import (
    VolatilitySizingParams,
    forecast_volatility_series,
    volatility_position_scale,
    volatility_stop_loss_pct,
)


def lagged_forecast(
    closes: pd.Series, index: pd.Index, params: VolatilitySizingParams
) -> pd.Series:
    """index の各足で使える見込み（前の足の終値までで作った値）を返す。"""
    return forecast_volatility_series(closes, params=params).shift(1).reindex(index)


class VolatilityOverlay:
    """1回のシミュレーションの中で、足ごとの量と損切りの線を返す。"""

    def __init__(
        self,
        size_params: Optional[VolatilitySizingParams],
        stop_params: Optional[VolatilitySizingParams],
        closes: pd.Series,
        index: pd.Index,
        execution: ExecutionModel,
    ):
        self._size_params = size_params if size_params and size_params.enabled else None
        self._stop_params = stop_params if stop_params and stop_params.enabled else None
        self._execution = execution
        self._size_forecast = (
            lagged_forecast(closes, index, self._size_params) if self._size_params else None
        )
        self._stop_forecast = (
            lagged_forecast(closes, index, self._stop_params) if self._stop_params else None
        )

    def scale_qty(self, qty: int, date: Any, cash: float, price: float, volume: float) -> int:
        """補正の前の量に倍率を掛け、単元に丸めて手元の現金までに収める。"""
        if self._size_params is None or self._size_forecast is None or qty <= 0:
            return qty
        scale = volatility_position_scale(
            _forecast_at(self._size_forecast, date), params=self._size_params
        )
        return cap_to_cash(
            self._execution.round_to_lot(int(qty * scale)), cash, price, volume, self._execution
        )

    def stop_loss_pct(self, fixed_stop_loss_pct: Optional[float], date: Any) -> Optional[float]:
        """建てる足で使える見込みから、その建玉の損切りの線を返す。"""
        if self._stop_params is None or self._stop_forecast is None:
            return fixed_stop_loss_pct
        return volatility_stop_loss_pct(
            fixed_stop_loss_pct, _forecast_at(self._stop_forecast, date), params=self._stop_params
        )


def _forecast_at(forecast: pd.Series, date: Any) -> Optional[float]:
    value = float(forecast.loc[date])
    return None if pd.isna(value) else value
