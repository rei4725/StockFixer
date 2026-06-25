"""ポートフォリオのパフォーマンス指標とレジーム別指標の計算。

Issue #511: 肥大化した portfolio.py を責務分割。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.data_port import get_backtest_data_port
from src.utils.logger import get_logger

from .simulation import _build_market_proxy_frame

logger = get_logger(__name__)


def _compute_portfolio_metrics(equity_df: pd.DataFrame, initial_cash: float) -> dict[str, Any]:
    """ポートフォリオ用メトリクスを計算する。"""
    if equity_df is None or equity_df.empty:
        return {}

    pv = equity_df["portfolio_value"]
    final = pv.iloc[-1]
    total_return = (final - initial_cash) / initial_cash

    daily_ret = pv.pct_change().dropna()
    sharpe = 0.0
    if len(daily_ret) >= 2 and daily_ret.std() > 0:
        sharpe = float(daily_ret.mean() / daily_ret.std() * math.sqrt(252))

    roll_max = pv.cummax()
    max_dd = float(((pv - roll_max) / roll_max).min())

    eq_pv = equity_df["equal_weight_value"]
    eq_return = (eq_pv.iloc[-1] - initial_cash) / initial_cash

    return {
        "final_cash": round(final, 2),
        "total_return": round(total_return, 6),
        "equal_weight_return": round(eq_return, 6),
        "alpha_vs_equal": round(total_return - eq_return, 6),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 6),
        "days": len(equity_df),
    }


def _attach_regime_metrics(
    equity_df: pd.DataFrame,
    close_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """ポートフォリオ日次損益にレジーム列を付与し、レジーム別メトリクスを返す。"""
    if equity_df is None or equity_df.empty or close_matrix is None or close_matrix.empty:
        return equity_df, {}

    proxy_df = _build_market_proxy_frame(close_matrix)
    if proxy_df.empty:
        return equity_df, {}

    regime_series = get_backtest_data_port().get_market_regime(proxy_df)
    if regime_series.empty:
        return equity_df, {}

    enriched = equity_df.copy()
    enriched["date"] = pd.to_datetime(enriched["date"])
    if not isinstance(regime_series.index, pd.DatetimeIndex):
        regime_series.index = pd.to_datetime(regime_series.index)

    min_regime_date = regime_series.index.min()
    enriched["regime"] = enriched["date"].map(
        lambda d: regime_series.asof(d) if d >= min_regime_date else None
    )
    regime_metrics = _compute_regime_metrics(enriched)
    enriched["date"] = enriched["date"].dt.strftime("%Y-%m-%d")
    return enriched, regime_metrics


def _compute_regime_metrics(equity_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {"all": _compute_regime_leg_metrics(equity_df)}
    if "regime" not in equity_df.columns:
        return results

    for label in ("bull", "bear", "range"):
        leg = equity_df[equity_df["regime"] == label]
        results[label] = _compute_regime_leg_metrics(leg)
    return results


def _compute_regime_leg_metrics(equity_leg: pd.DataFrame) -> dict[str, Any]:
    if equity_leg is None or equity_leg.empty:
        return {
            "days": 0,
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "hit_rate": 0.0,
            "max_drawdown": 0.0,
        }

    portfolio_values = pd.to_numeric(equity_leg["portfolio_value"], errors="coerce").dropna()
    daily_returns = portfolio_values.pct_change().dropna()
    daily_return_values = daily_returns.to_numpy(dtype=float, copy=False)
    total_return = (
        float(np.prod(1.0 + daily_return_values) - 1.0) if len(daily_return_values) else 0.0
    )
    sharpe_ratio = 0.0
    if len(daily_return_values) >= 2:
        daily_std = float(np.std(daily_return_values, ddof=1))
        if daily_std > 0:
            daily_mean = float(np.mean(daily_return_values))
            sharpe_ratio = float(daily_mean / daily_std * math.sqrt(252))

    hit_rate = float(np.mean(daily_return_values > 0)) if len(daily_return_values) else 0.0
    curve = pd.Series(np.cumprod(1.0 + daily_return_values), index=daily_returns.index)
    max_drawdown = 0.0
    if not curve.empty:
        roll_max = curve.cummax()
        max_drawdown = float(((curve - roll_max) / roll_max).min())

    return {
        "days": int(len(equity_leg)),
        "total_return": round(total_return, 6),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "hit_rate": round(hit_rate, 4),
        "max_drawdown": round(max_drawdown, 6),
    }
