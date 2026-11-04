"""値動きの見込みと、発注の量・損切りの線の補正のテスト（#1528）。"""

import math

import numpy as np
import pandas as pd
import pytest

from config.settings import MAX_POSITION_RATE
from src.domain.volatility_sizing import (
    DEFAULT_VOLATILITY_SIZING,
    VolatilitySizingParams,
    forecast_volatility,
    forecast_volatility_series,
    volatility_adjusted_amount,
    volatility_position_scale,
    volatility_stop_loss_pct,
)

ON = VolatilitySizingParams(enabled=True)


def _alternating_closes(daily_move: float, n: int = 60) -> list[float]:
    """毎日 ±daily_move の対数リターンで動く終値。"""
    returns = [daily_move if i % 2 == 0 else -daily_move for i in range(n)]
    return [float(p) for p in 1000.0 * np.exp(np.cumsum([0.0, *returns]))]


def _random_closes(n: int = 120) -> pd.Series:
    rng = np.random.default_rng(1528)
    return pd.Series(1000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, n))))


class TestForecastVolatility:
    def test_constant_daily_move_is_recovered(self) -> None:
        assert forecast_volatility(_alternating_closes(0.03)) == pytest.approx(0.03)

    def test_value_at_t_equals_value_from_closes_up_to_t(self) -> None:
        closes = _random_closes()
        full = forecast_volatility_series(closes)
        for t in (30, 60, 119):
            truncated = forecast_volatility(closes.iloc[: t + 1])
            assert truncated == pytest.approx(float(full.iloc[t]))

    def test_changing_future_closes_does_not_change_past_forecasts(self) -> None:
        closes = _random_closes()
        changed = closes.copy()
        changed.iloc[70:] = changed.iloc[70:] * 3.0
        before = forecast_volatility_series(closes).iloc[:70]
        after = forecast_volatility_series(changed).iloc[:70]
        pd.testing.assert_series_equal(before, after)

    def test_keeps_index_of_input(self) -> None:
        closes = _random_closes(30)
        closes.index = pd.date_range("2026-01-01", periods=30)
        assert forecast_volatility_series(closes).index.equals(closes.index)

    def test_exactly_enough_observations_gives_value(self) -> None:
        closes = _alternating_closes(0.02, n=DEFAULT_VOLATILITY_SIZING.min_observations)
        assert forecast_volatility(closes) == pytest.approx(0.02)

    def test_one_observation_short_gives_none(self) -> None:
        closes = _alternating_closes(0.02, n=DEFAULT_VOLATILITY_SIZING.min_observations - 1)
        assert forecast_volatility(closes) is None

    @pytest.mark.parametrize(
        "closes",
        [
            [],
            [1000.0],
            [float("nan")] * 40,
            [0.0] * 40,
            [float("inf")] * 40,
        ],
    )
    def test_unusable_closes_give_none(self, closes: list[float]) -> None:
        assert forecast_volatility(closes) is None

    def test_flat_price_gives_none(self) -> None:
        assert forecast_volatility([1000.0] * 40) is None

    def test_bad_close_is_not_bridged_into_a_return(self) -> None:
        closes = _alternating_closes(0.02)
        closes[-10] = 0.0  # 0 を欠けとして扱い、前後の大きなリターンを作らない
        assert forecast_volatility(closes) == pytest.approx(0.02)


class TestPositionSize:
    def test_double_forecast_halves_amount(self) -> None:
        base = volatility_adjusted_amount(50_000.0, 1e9, 0.02, max_position_rate=0.1, params=ON)
        doubled = volatility_adjusted_amount(50_000.0, 1e9, 0.04, max_position_rate=0.1, params=ON)
        assert doubled == pytest.approx(base / 2)

    def test_scale_upper_bound(self) -> None:
        assert volatility_position_scale(0.001, params=ON) == ON.max_scale

    def test_scale_lower_bound(self) -> None:
        assert volatility_position_scale(1.0, params=ON) == ON.min_scale

    def test_amount_respects_max_position_rate(self) -> None:
        balance = 1_000_000.0
        amount = volatility_adjusted_amount(
            90_000.0, balance, 0.005, max_position_rate=MAX_POSITION_RATE, params=ON
        )
        assert amount == pytest.approx(balance * MAX_POSITION_RATE)

    def test_amount_is_not_negative(self) -> None:
        assert volatility_adjusted_amount(10.0, -100.0, 0.02, max_position_rate=0.1, params=ON) == 0

    @pytest.mark.parametrize("forecast", [None, 0.0, -0.01, math.nan, math.inf])
    def test_unusable_forecast_scale_is_one(self, forecast: float | None) -> None:
        assert volatility_position_scale(forecast, params=ON) == 1.0

    def test_unusable_forecast_still_respects_cap(self) -> None:
        amount = volatility_adjusted_amount(
            200_000.0, 1_000_000.0, None, max_position_rate=0.1, params=ON
        )
        assert amount == pytest.approx(100_000.0)

    def test_off_returns_amount_unchanged(self) -> None:
        # 上限も掛けない。off の発注の結果を1円も変えないため
        assert volatility_adjusted_amount(123_456.0, 1.0, 0.5, max_position_rate=0.1) == 123_456.0

    def test_off_scale_is_one(self) -> None:
        assert volatility_position_scale(0.5) == 1.0


class TestStopLoss:
    def test_stop_is_k_times_forecast(self) -> None:
        params = VolatilitySizingParams(enabled=True, stop_volatility_multiple=2.5)
        assert volatility_stop_loss_pct(0.08, 0.02, params=params) == pytest.approx(0.05)

    def test_stop_upper_bound(self) -> None:
        assert volatility_stop_loss_pct(0.08, 0.9, params=ON) == ON.max_stop_loss_pct

    def test_off_keeps_fixed_rate(self) -> None:
        assert volatility_stop_loss_pct(0.08, 0.02) == 0.08

    def test_off_keeps_none(self) -> None:
        assert volatility_stop_loss_pct(None, 0.02) is None

    @pytest.mark.parametrize("forecast", [None, 0.0, -0.01, math.nan, math.inf])
    def test_unusable_forecast_keeps_fixed_rate(self, forecast: float | None) -> None:
        assert volatility_stop_loss_pct(0.08, forecast, params=ON) == 0.08


class TestParams:
    def test_default_is_off(self) -> None:
        assert DEFAULT_VOLATILITY_SIZING.enabled is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"ewma_lambda": 1.0},
            {"ewma_lambda": 0.0},
            {"min_observations": 0},
            {"target_daily_volatility": 0.0},
            {"min_scale": 0.0},
            {"min_scale": 3.0},
            {"stop_volatility_multiple": 0.0},
            {"max_stop_loss_pct": 1.0},
        ],
    )
    def test_invalid_params_are_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            VolatilitySizingParams(**kwargs)
