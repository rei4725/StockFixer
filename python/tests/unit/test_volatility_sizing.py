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
    def test_default_is_sample_std_of_last_20_daily_log_returns(self) -> None:
        closes = _random_closes()
        log_returns = np.diff(np.log(closes.to_numpy()))
        expected = float(np.std(log_returns[-20:], ddof=1))
        assert forecast_volatility(closes) == pytest.approx(expected)

    def test_series_matches_std_of_trailing_window_at_each_t(self) -> None:
        closes = _random_closes()
        log_returns = np.diff(np.log(closes.to_numpy()))
        series = forecast_volatility_series(closes)
        for t in (20, 21, 60, 119):
            expected = float(np.std(log_returns[t - 20 : t], ddof=1))
            assert float(series.iloc[t]) == pytest.approx(expected)

    def test_close_older_than_the_window_does_not_change_forecast(self) -> None:
        closes = _random_closes()
        changed = closes.copy()
        changed.iloc[:-21] = changed.iloc[:-21] * 3.0
        assert forecast_volatility(changed) == pytest.approx(forecast_volatility(closes))

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
        expected = float(np.std(np.diff(np.log(closes)), ddof=1))
        assert forecast_volatility(closes) == pytest.approx(expected)

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

    def test_flat_tail_after_moves_gives_none(self) -> None:
        closes = _alternating_closes(0.02) + [1000.0] * 21
        assert forecast_volatility(closes) is None

    @pytest.mark.parametrize("bad_close", [0.0, -1.0, float("nan"), float("inf")])
    @pytest.mark.parametrize("position", [-1, -10, -21])
    def test_bad_close_in_window_gives_none(self, bad_close: float, position: int) -> None:
        closes = _alternating_closes(0.02)
        closes[position] = bad_close
        assert forecast_volatility(closes) is None

    def test_bad_close_out_of_window_is_ignored(self) -> None:
        closes = _alternating_closes(0.02)
        expected = forecast_volatility(closes)
        closes[-22] = 0.0
        assert forecast_volatility(closes) == pytest.approx(expected)

    def test_bad_close_is_not_bridged_into_a_return(self) -> None:
        # 要る数を窓より減らすと欠けを除いて計算する。欠けの前後のリターンは作らない
        params = VolatilitySizingParams(min_observations=15)
        closes = _alternating_closes(0.02)
        log_returns = np.diff(np.log(closes))
        closes[-10] = 0.0
        kept = np.delete(log_returns[-20:], [10, 11])
        expected = float(np.std(kept, ddof=1))
        assert forecast_volatility(closes, params=params) == pytest.approx(expected)


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

    def test_default_window_is_20_days(self) -> None:
        assert DEFAULT_VOLATILITY_SIZING.volatility_window == 20
        assert DEFAULT_VOLATILITY_SIZING.min_observations == 20

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"volatility_window": 1},
            {"min_observations": 0},
            {"min_observations": 21},
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
