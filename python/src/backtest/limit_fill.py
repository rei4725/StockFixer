"""シグナル約定に本番と同じ指値の判定を当てる（#1418）。

本番の発注は、平均出来高が少ないか値幅が広い銘柄を指値で出す（trading の
_choose_order_params）。Backtester もその銘柄では、翌バーが指値に届いた日だけ指値の
値段で約定させ、届かなければ約定させない。判定と定数は本番と同じものを使う。
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from config.settings import (
    LIMIT_ORDER_AVG_VOLUME_THRESHOLD,
    LIMIT_ORDER_LOOKBACK_DAYS,
    LIMIT_ORDER_PRICE_BUFFER,
    LIMIT_ORDER_SPREAD_PROXY_THRESHOLD,
)
from src.domain.limit_order_rules import limit_order_fills, plan_limit_order


def _optional(value: Any) -> Optional[float]:
    return None if pd.isna(value) else float(value)


class LimitFillModel:
    """各バーのシグナル約定の値段を決める。enabled が False なら常に exec_price を返す。

    平均出来高と値幅は、本番の直近 LIMIT_ORDER_LOOKBACK_DAYS 本の平均と同じく、
    約定の日を含まない前のバーまでで数える（先読みしない）。指値の基準の現在値は
    前のバーの終値（シグナルを出した引けの値段）とする。
    """

    def __init__(self, df: pd.DataFrame, close_col: str, market: str, enabled: bool) -> None:
        self.market = market
        self.enabled = enabled
        close = pd.to_numeric(df[close_col], errors="coerce")
        self._close = close
        self._high = pd.to_numeric(df["High"], errors="coerce") if "High" in df.columns else close
        self._low = pd.to_numeric(df["Low"], errors="coerce") if "Low" in df.columns else close
        self._ref_price = close.shift(1)
        window = LIMIT_ORDER_LOOKBACK_DAYS
        self._avg_volume: Optional[pd.Series] = None
        if "Volume" in df.columns:
            volume = pd.to_numeric(df["Volume"], errors="coerce")
            self._avg_volume = volume.rolling(window, min_periods=1).mean().shift(1)
        self._spread: Optional[pd.Series] = None
        if {"High", "Low"}.issubset(df.columns):
            ratio = (self._high - self._low) / close.where(close != 0)
            self._spread = ratio.rolling(window, min_periods=1).mean().shift(1)

    def fill_price(self, date: Any, is_buy_side: bool, exec_price: float) -> Optional[float]:
        """約定の値段を返す。指値に届かず約定しないときは None。"""
        if not self.enabled:
            return exec_price
        ref_price = _optional(self._ref_price.loc[date])
        if ref_price is None or ref_price <= 0:
            return exec_price
        plan = plan_limit_order(
            self.market,
            is_buy_side,
            ref_price,
            _optional(self._avg_volume.loc[date]) if self._avg_volume is not None else None,
            _optional(self._spread.loc[date]) if self._spread is not None else None,
            LIMIT_ORDER_AVG_VOLUME_THRESHOLD,
            LIMIT_ORDER_SPREAD_PROXY_THRESHOLD,
            LIMIT_ORDER_PRICE_BUFFER,
        )
        if not plan.is_limit:
            return exec_price
        bar = (self._high.loc[date], self._low.loc[date], self._close.loc[date])
        if any(pd.isna(v) for v in bar):
            return None
        if not limit_order_fills(plan, is_buy_side, float(bar[0]), float(bar[1]), float(bar[2])):
            return None
        return plan.limit_price
