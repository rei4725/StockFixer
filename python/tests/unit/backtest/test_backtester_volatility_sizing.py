"""Backtester の値動きの見込みの量と損切りの線（#1547）。"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest

from src.backtest.backtester import Backtester
from src.domain.volatility_sizing import VolatilitySizingParams, forecast_volatility_series

_ENTRY_BAR = 30
_ON = VolatilitySizingParams(enabled=True, target_daily_volatility=0.01)


def _backtester(**overrides: Any) -> Backtester:
    kwargs: dict[str, Any] = {
        "model_manager": None,
        "signal_generator": None,
        "data_loader": None,
        "start_date": None,
        "end_date": None,
        "market": "us",
        "symbol": "TEST",
        "initial_cash": 1_000_000,
        "position_sizing": "fixed",
        "position_fraction": 0.25,
        "limit_order_fill": False,
    }
    kwargs.update(overrides)
    return Backtester(**kwargs)


def _prices(daily_move: float, bars: int = 60) -> pd.DataFrame:
    """100 の上下を日ごとに daily_move だけ行き来する値段。"""
    closes = [100.0 * (1.0 + daily_move * (1 if i % 2 else -1)) for i in range(bars)]
    index = pd.date_range("2025-01-01", periods=bars, freq="B")
    return pd.DataFrame({"Open": closes, "Close": closes, "Volume": 1_000_000.0}, index=index)


def _buy_signal(df: pd.DataFrame, bar: int = _ENTRY_BAR) -> pd.Series:
    signal = pd.Series(0, index=df.index)
    signal.iloc[bar] = 1
    return signal


def _buy_qty(bt: Backtester, df: pd.DataFrame, closes: Optional[pd.Series] = None) -> int:
    result, _ = bt.simulate_trading(df, _buy_signal(df), volatility_closes=closes)
    return int(result[result["action"] == "buy"]["qty"].iloc[0])


def test_off_matches_the_current_result() -> None:
    """既定（None）と enabled=False は、同じ取引と同じ成績になる"""
    df = _prices(0.02)
    off = VolatilitySizingParams(enabled=False)
    signal = _buy_signal(df)
    base_log, base_metrics = _backtester(stop_loss_pct=0.03).simulate_trading(df, signal)
    off_log, off_metrics = _backtester(
        stop_loss_pct=0.03, volatility_size_params=off, volatility_stop_params=off
    ).simulate_trading(df, signal)

    pd.testing.assert_frame_equal(base_log, off_log)
    assert base_metrics == off_metrics
    # 固定の線のまま: 1回目の下げ（-4%）で刈られ、量は 25% の分
    assert base_log["action"].tolist() == ["buy", "stop_loss"]
    assert base_log["qty"].iloc[0] == int(1_000_000 * 0.25 // base_log["price"].iloc[0])


def test_on_twice_the_forecast_gives_about_half_the_quantity() -> None:
    """見込みの値動きが2倍の銘柄の量が、およそ半分になる"""
    calm = _prices(0.005)
    rough = _prices(0.01)
    bt = _backtester(volatility_size_params=_ON)

    calm_value = _buy_qty(bt, calm) * calm["Close"].iloc[_ENTRY_BAR + 1]
    rough_value = _buy_qty(bt, rough) * rough["Close"].iloc[_ENTRY_BAR + 1]

    assert rough_value / calm_value == pytest.approx(0.5, rel=0.03)


def test_on_scale_follows_the_target() -> None:
    """見込みが目標と同じくらいの銘柄は、量が補正の前とおよそ同じになる"""
    df = _prices(0.01)
    base = _buy_qty(_backtester(), df)
    scaled = _buy_qty(_backtester(volatility_size_params=_ON), df)
    forecast = forecast_volatility_series(df["Close"], params=_ON).iloc[_ENTRY_BAR]

    assert scaled == pytest.approx(base * 0.01 / forecast, abs=1.0)


def test_on_stop_line_is_k_times_the_forecast() -> None:
    """損切りの線が、建てたときの見込みの k 倍になる"""
    df = _prices(0.01)
    entry_bar = _ENTRY_BAR + 1  # 見込みは約定の足の前の足まで
    forecast = float(forecast_volatility_series(df["Close"]).iloc[entry_bar - 1])
    entry = df["Open"].iloc[entry_bar]
    line = 3.0 * forecast
    tail = len(df) - entry_bar - 1

    def run(drop: float) -> list[str]:
        moved = df.copy()
        closes = moved.columns.get_loc("Close")
        opens = moved.columns.get_loc("Open")
        moved.iloc[entry_bar:, closes] = entry
        moved.iloc[entry_bar:, opens] = entry
        moved.iloc[entry_bar + 1, closes] = entry * (1.0 - drop)
        assert tail > 2
        params = VolatilitySizingParams(enabled=True, stop_volatility_multiple=3.0)
        bt = _backtester(stop_loss_pct=0.50, volatility_stop_params=params)
        result, _ = bt.simulate_trading(moved, _buy_signal(moved), volatility_closes=df["Close"])
        return result["action"].tolist()

    assert "stop_loss" not in run(line * 0.95)
    assert "stop_loss" in run(line * 1.05)


@pytest.mark.parametrize("lag", [0, 1])
def test_forecast_does_not_use_the_close_of_the_bar(lag: int) -> None:
    """約定の足の終値を変えても量が変わらない（先読みが無い）"""
    df = _prices(0.01)
    fill_bar = _ENTRY_BAR + lag
    moved = df.copy()
    moved.iloc[fill_bar, moved.columns.get_loc("Close")] *= 1.5
    bt = _backtester(execution_lag=lag, volatility_size_params=_ON)
    expected = _ON.target_daily_volatility / float(
        forecast_volatility_series(df["Close"], params=_ON).iloc[fill_bar - 1]
    )

    qty = _buy_qty(bt, moved)
    base = _buy_qty(_backtester(execution_lag=lag), moved)

    assert qty == pytest.approx(base * expected, abs=1.0)


def test_warmup_closes_give_a_forecast_from_the_first_bar() -> None:
    """渡した終値に前の足があれば、シミュレーションの最初の足から見込みが使える"""
    df = _prices(0.01)
    tail = df.iloc[_ENTRY_BAR:]
    bt = _backtester(volatility_size_params=_ON)
    signal = pd.Series(0, index=tail.index)
    signal.iloc[0] = 1

    without, _ = bt.simulate_trading(tail, signal)
    warmed, _ = bt.simulate_trading(tail, signal, volatility_closes=df["Close"])

    assert warmed["qty"].iloc[0] != without["qty"].iloc[0]
    fixed, _ = _backtester().simulate_trading(tail, signal)
    assert without["qty"].iloc[0] == fixed["qty"].iloc[0]


def test_short_history_falls_back_to_the_fixed_settings() -> None:
    """見込みが出ない足では、量も損切りの線も固定のまま"""
    df = _prices(0.02, bars=10)
    signal = _buy_signal(df, bar=2)
    fixed_log, _ = _backtester(stop_loss_pct=0.03).simulate_trading(df, signal)
    on_log, _ = _backtester(
        stop_loss_pct=0.03, volatility_size_params=_ON, volatility_stop_params=_ON
    ).simulate_trading(df, signal)

    pd.testing.assert_frame_equal(fixed_log, on_log)


def test_jp_lot_rounding_applies_to_the_scaled_quantity() -> None:
    """補正の後の量も単元株に丸める"""
    df = _prices(0.01)
    qty = _buy_qty(_backtester(market="jp", volatility_size_params=_ON), df)

    assert qty > 0
    assert np.mod(qty, 100) == 0


def test_scaled_up_quantity_stays_within_cash() -> None:
    """量を増やしても、手元の現金を超えて買わない"""
    df = _prices(0.002)
    bt = _backtester(position_fraction=0.9, fee_rate=0.001, volatility_size_params=_ON)
    result, _ = bt.simulate_trading(df, _buy_signal(df))

    assert result["cash"].iloc[0] >= 0.0
