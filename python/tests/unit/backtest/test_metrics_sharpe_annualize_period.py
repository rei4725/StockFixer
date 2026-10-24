"""compute_metrics の Sharpe 年率化が、資産曲線の期間で取引頻度を出すことのテスト（#1022）"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.backtest.metrics.core import compute_metrics

_PNLS = [5000.0, -1000.0, 6000.0]


def _clustered_trade_log() -> pd.DataFrame:
    """3回の往復の取引が、2024-03-01 からの8日の間に寄ったログ。"""
    dates = pd.to_datetime(
        [
            "2024-03-01",
            "2024-03-02",
            "2024-03-04",
            "2024-03-05",
            "2024-03-07",
            "2024-03-09",
        ]
    )
    return pd.DataFrame(
        {
            "date": dates,
            "action": ["buy", "sell", "buy", "sell", "buy", "sell"],
            "price": [1000.0, 1050.0, 1050.0, 1040.0, 1040.0, 1100.0],
            "qty": [100, 100, 100, 100, 100, 100],
            "cash": [900_000.0, 1_005_000.0, 900_000.0, 1_004_000.0, 900_000.0, 1_010_000.0],
        }
    )


def _expected_sharpe(trades_per_year: float) -> float:
    arr = np.array(_PNLS)
    return float(arr.mean() / arr.std(ddof=1)) * math.sqrt(trades_per_year)


def test_sharpe_uses_equity_series_period_when_given() -> None:
    """資産曲線が1年に渡り取引が一部に寄るとき、年率化は資産曲線の期間で行う。"""
    index = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    equity = pd.Series(1_000_000.0, index=index)

    result = compute_metrics(_clustered_trade_log(), 1_000_000.0, equity_series=equity)

    span_days = (index[-1] - index[0]).days
    expected = _expected_sharpe(3 / (span_days / 365.25))
    assert result["sharpe_ratio"] == round(expected, 4)


def test_sharpe_falls_back_to_trade_dates_without_equity_series() -> None:
    """equity_series を渡さないときは、取引の日付の差で年率化する（今の値のまま）。"""
    result = compute_metrics(_clustered_trade_log(), 1_000_000.0)

    expected = _expected_sharpe(3 / (8 / 365.25))
    assert result["sharpe_ratio"] == round(expected, 4)


def test_sharpe_falls_back_when_equity_series_is_empty() -> None:
    """空の資産曲線は渡されないのと同じに扱う。"""
    result = compute_metrics(
        _clustered_trade_log(), 1_000_000.0, equity_series=pd.Series(dtype=float)
    )

    expected = _expected_sharpe(3 / (8 / 365.25))
    assert result["sharpe_ratio"] == round(expected, 4)
